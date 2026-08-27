use smallvec::SmallVec;
use ty_module_resolver::Module;
use ty_python_core::definition::{Definition, DefinitionKind, DefinitionState};
use ty_python_core::scope::ScopeId;
use ty_python_core::{
    BindingWithConstraintsIterator, Program, ProgramFile, global_scope, place_table, use_def_map,
};

use crate::Db;
use crate::place::{
    Place, builtins_module_scope, class_body_implicit_symbol, implicit_builtins_symbol,
    loop_header_reachability, module_type_implicit_global_symbol,
};
use crate::place_load::{ImplicitPlaceLoad, PlaceLoadSource, PlaceLoadSourceKind};
use crate::reachability::ReachabilityConstraintsExtension;
use crate::types::ProgramEnvironment;

/// Returns the definitions that may supply the value for a module global at the end of its scope.
pub(crate) fn definitions_for_module_global<'db>(
    db: &'db dyn Db,
    program: Program<'db>,
    module: Module<'db>,
    name: &str,
) -> Option<DefinitionResolution<'db>> {
    let file = ProgramFile::new(db, module.file(db)?, program);
    let scope = global_scope(db, file);
    let symbol = place_table(db, scope).symbol_id(name)?;

    Some(DefinitionResolution::from_bindings(
        db,
        use_def_map(db, scope).end_of_scope_symbol_bindings(symbol),
    ))
}

/// Records the definitions that can supply a name's value and the limits of that resolution.
///
/// A consumer needs more than the definitions to decide whether it can rewrite a name safely.
/// Resolution also tracks whether values lack explicit definitions, whether the name can be
/// deleted, and whether lookup crosses a `global` or `nonlocal` declaration.
#[derive(Debug, Clone, Eq, PartialEq, get_size2::GetSize, salsa::SalsaValue)]
pub struct DefinitionResolution<'db> {
    definitions: SmallVec<[Definition<'db>; 2]>,
    is_complete: bool,
    may_be_deleted: bool,
    crosses_scope_declaration: bool,
}

#[allow(
    dead_code,
    reason = "definition-resolution metadata is retained for IDE consumers"
)]
impl<'db> DefinitionResolution<'db> {
    /// Returns the definitions found by name resolution.
    pub fn definitions(&self) -> &[Definition<'db>] {
        &self.definitions
    }

    /// Returns whether every possible result is represented by a definition.
    ///
    /// Implicit builtin values are incomplete because no explicit import connects the name
    /// to their definitions. A complete resolution can still leave a name possibly unbound.
    pub fn is_complete(&self) -> bool {
        self.is_complete
    }

    /// Returns whether a reachable deletion may leave the value unbound.
    pub fn may_be_deleted(&self) -> bool {
        self.may_be_deleted
    }

    /// Returns whether resolution crosses a `global` or `nonlocal` declaration.
    pub fn crosses_scope_declaration(&self) -> bool {
        self.crosses_scope_declaration
    }

    /// Replaces each definition with its projected definitions.
    ///
    /// The result is incomplete if any definition has no projection.
    pub(crate) fn project_definitions<I>(
        mut self,
        mut project: impl FnMut(Definition<'db>) -> I,
    ) -> Self
    where
        I: IntoIterator<Item = Definition<'db>>,
    {
        let definitions = std::mem::take(&mut self.definitions);

        for definition in definitions {
            let mut has_projection = false;
            for projected in project(definition) {
                has_projection = true;
                self.push_definition(projected);
            }
            self.is_complete &= has_projection;
        }

        self
    }

    fn from_place_load_source(
        db: &'db dyn Db,
        environment: &ProgramEnvironment<'db>,
        scope: ScopeId<'db>,
        source: &PlaceLoadSource<'db>,
    ) -> Self {
        match &source.kind {
            PlaceLoadSourceKind::Bindings(bindings) => Self::from_bindings(db, bindings.clone()),
            PlaceLoadSourceKind::DefinitionsFromOwningScope { scope, id } => {
                Self::from_bindings(db, use_def_map(db, *scope).reachable_bindings(*id))
            }
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::ExplicitGlobalSymbol {
                file,
                name,
            }) => {
                let scope = global_scope(db, *file);
                let Some(symbol) = place_table(db, scope).symbol_id(name) else {
                    return Self {
                        definitions: SmallVec::new(),
                        is_complete: true,
                        may_be_deleted: false,
                        crosses_scope_declaration: false,
                    };
                };
                Self::from_bindings(db, use_def_map(db, scope).reachable_symbol_bindings(symbol))
            }
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::DunderClass(class_def)) => {
                let mut resolution = Self {
                    definitions: SmallVec::new(),
                    is_complete: true,
                    may_be_deleted: false,
                    crosses_scope_declaration: false,
                };
                resolution.push_definition(*class_def);
                resolution
            }
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::ClassBodySymbol(name)) => {
                Self::from_place_without_definition(
                    class_body_implicit_symbol(db, environment, name).place,
                )
            }
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::ModuleImplicitGlobal {
                file,
                name,
            }) => Self::from_place_without_definition(
                module_type_implicit_global_symbol(db, *file, name).place,
            ),
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::Builtin(name)) => {
                Self::from_builtin(db, environment, scope, name)
            }
        }
    }

    fn from_builtin(
        db: &'db dyn Db,
        environment: &ProgramEnvironment<'db>,
        scope: ScopeId<'db>,
        name: &str,
    ) -> Self {
        if Some(scope) == builtins_module_scope(db, environment) {
            // A missing name in `builtins` cannot fall back to the module that is currently being
            // resolved. Treating it as undefined also avoids a recursive semantic query.
            return Self::from_place_without_definition(Place::Undefined);
        }

        // Builtin values have no explicit import for a consumer to follow. Record whether a
        // value exists without collecting definitions that consumers cannot use.
        Self::from_place_without_definition(implicit_builtins_symbol(db, environment, name).place)
    }

    /// Resolves the reachable definitions supplied by the given bindings.
    pub(crate) fn from_bindings(
        db: &'db dyn Db,
        mut bindings: BindingWithConstraintsIterator<'db, 'db>,
    ) -> Self {
        let mut resolution = Self {
            definitions: SmallVec::new(),
            is_complete: true,
            may_be_deleted: false,
            crosses_scope_declaration: false,
        };

        while let Some(binding) = bindings.next() {
            let reachability = bindings.reachability_constraints().evaluate(
                db,
                bindings.predicates(),
                binding.reachability_constraint,
            );
            if reachability.is_always_false() {
                continue;
            }

            match binding.binding {
                DefinitionState::Defined(definition) => {
                    if matches!(definition.kind(db), DefinitionKind::LoopHeader(_)) {
                        let deleted_reachability =
                            loop_header_reachability(db, definition).deleted_reachability;
                        let may_be_deleted = !deleted_reachability.is_always_false();
                        resolution.may_be_deleted |= may_be_deleted;
                    }
                    resolution.push_definition(definition);
                }
                DefinitionState::Deleted => {
                    let may_be_deleted = reachability.may_be_true();
                    resolution.may_be_deleted |= may_be_deleted;
                }
                DefinitionState::Undefined => {}
            }
        }

        resolution
    }

    fn push_definition(&mut self, definition: Definition<'db>) {
        if !self.definitions.contains(&definition) {
            self.definitions.push(definition);
        }
    }

    fn from_place_without_definition(place: Place<'db>) -> Self {
        Self {
            definitions: SmallVec::new(),
            is_complete: place.is_undefined(),
            may_be_deleted: false,
            crosses_scope_declaration: false,
        }
    }

    fn extend(&mut self, other: Self) {
        for definition in other.definitions {
            if !self.definitions.contains(&definition) {
                self.definitions.push(definition);
            }
        }
        self.is_complete &= other.is_complete;
        self.may_be_deleted |= other.may_be_deleted;
        self.crosses_scope_declaration |= other.crosses_scope_declaration;
    }
}

/// Accumulates definition information while type inference resolves a place load.
pub(crate) struct DefinitionResolutionBuilder<'db> {
    resolution: DefinitionResolution<'db>,
}

impl<'db> DefinitionResolutionBuilder<'db> {
    /// Starts recording a name load before any of its sources have been visited.
    pub(crate) fn new() -> Self {
        Self {
            resolution: DefinitionResolution {
                definitions: SmallVec::new(),
                is_complete: true,
                may_be_deleted: false,
                crosses_scope_declaration: false,
            },
        }
    }

    /// Records the reachable definitions and limitations of a source visited by inference.
    pub(crate) fn add_source(
        &mut self,
        db: &'db dyn Db,
        environment: &ProgramEnvironment<'db>,
        scope: ScopeId<'db>,
        source: &PlaceLoadSource<'db>,
    ) {
        let source_resolution =
            DefinitionResolution::from_place_load_source(db, environment, scope, source);
        self.resolution.extend(source_resolution);
    }

    /// Marks a load whose possible values are not fully represented by the recorded definitions.
    pub(crate) fn mark_incomplete(&mut self) {
        self.resolution.is_complete = false;
    }

    /// Finishes the record with the scope-declaration state from name resolution.
    pub(crate) fn finish(mut self, crosses_scope_declaration: bool) -> DefinitionResolution<'db> {
        self.resolution.crosses_scope_declaration |= crosses_scope_declaration;
        self.resolution.definitions.shrink_to_fit();
        self.resolution
    }
}

#[cfg(test)]
mod tests {
    use ruff_db::files::system_path_to_file;
    use ty_python_core::ProgramFile;

    use super::definitions_for_module_global;
    use crate::SemanticModel;
    use crate::db::tests::TestDbBuilder;

    #[test]
    fn definitions_for_module_global_retains_conditional_definitions() {
        let db = TestDbBuilder::new()
            .with_file(
                "/src/test.py",
                r#"
if flag:
    value = 1
else:
    value = 2
"#,
            )
            .build()
            .expect("valid TestDb setup");
        let file = system_path_to_file(&db, "/src/test.py").expect("test file should exist");
        let program = db.program_environment().program(&db);
        let model = SemanticModel::new(&db, ProgramFile::new(&db, file, program));
        let module = model
            .resolve_module(Some("test"), 0)
            .expect("test module should resolve");

        let resolution = definitions_for_module_global(&db, program, module, "value")
            .expect("module global should exist");

        assert_eq!(resolution.definitions().len(), 2);
        assert!(resolution.is_complete());
    }
}
