//! Retrieves reaching source definitions for IDE features.
//!
//! Name queries reuse recorded inference results. Callers must enable recording in the database.

#![allow(
    dead_code,
    reason = "source-backed definition resolution is retained for IDE consumers"
)]

use ruff_python_ast as ast;
use ty_module_resolver::Module;

use crate::SemanticModel;
use crate::place::definitions::{DefinitionResolution, definitions_for_module_global};
use crate::types::name_load_resolutions_from_inference;

use super::user_visible_definitions;

impl<'db> SemanticModel<'db> {
    /// Returns the source definitions that may supply the value read by `name`.
    ///
    /// Infers the enclosing scope as needed, reusing its cached results for subsequent names.
    /// For names in string annotations, use the model from [`Self::enter_string_annotation`].
    /// Returns `None` if recording is disabled or inference did not visit the name.
    /// A result does not guarantee the name is bound; inspect its resolution flags before editing.
    pub fn reaching_definitions(&self, name: &ast::ExprName) -> Option<DefinitionResolution<'db>> {
        let scope = self
            .scope(name.into())?
            .to_scope_id(self.db(), self.program_file());
        let resolutions = name_load_resolutions_from_inference(self.db(), scope)?;
        let resolution = resolutions.get(&ast::ExprRef::Name(name).into())?;
        Some(source_backed_resolution(self.db(), resolution.clone()))
    }

    /// Returns the source definitions that may supply a module global at the end of its scope.
    ///
    /// Returns `None` if the module has no file or the name has no entry in its symbol table.
    /// A result does not guarantee the name is bound; inspect its resolution flags before editing.
    pub fn definitions_for_module_global(
        &self,
        module: Module<'db>,
        name: &str,
    ) -> Option<DefinitionResolution<'db>> {
        definitions_for_module_global(self.db(), self.program(), module, name)
            .map(|resolution| source_backed_resolution(self.db(), resolution))
    }
}

/// Replaces synthetic bindings with their user-visible definitions while preserving resolution flags.
///
/// If any binding has no user-visible definition, the result is marked incomplete so refactoring
/// consumers do not mistake a partial set of definitions for the full set.
pub(super) fn source_backed_resolution<'db>(
    db: &'db dyn crate::Db,
    resolution: DefinitionResolution<'db>,
) -> DefinitionResolution<'db> {
    resolution.project_definitions(|definition| user_visible_definitions(db, [definition]))
}

#[cfg(test)]
mod tests {
    use ruff_db::files::system_path_to_file;
    use ruff_db::parsed::{ParsedModuleRef, parsed_module};
    use ruff_python_ast::visitor::{Visitor, walk_expr};
    use ruff_python_ast::{self as ast};
    use ruff_text_size::Ranged;
    use ty_python_core::ProgramFile;

    use crate::db::tests::TestDbBuilder;
    use crate::{PlaceLoadRecordingMode, SemanticModel};

    #[test]
    fn observed_loads_inside_string_annotation_are_distinct() {
        let source = r#"
import first
import second
annotation: "tuple[first.C, second.C]"
"#;
        let path = "/src/test.py";
        let db = TestDbBuilder::new()
            .with_place_load_recording_mode(PlaceLoadRecordingMode::Enabled)
            .with_file(path, source)
            .build()
            .expect("valid test database");
        let file = system_path_to_file(&db, path).expect("test file should exist");
        let file = ProgramFile::new(&db, file, db.program_environment().program(&db));
        let module = parsed_module(&db, file.python_file(&db)).load(&db);
        let assignment = module
            .syntax()
            .body
            .last()
            .and_then(ast::Stmt::as_ann_assign_stmt)
            .expect("last statement should be an annotated assignment");
        let annotation = assignment
            .annotation
            .as_string_literal_expr()
            .expect("assignment annotation should be a string literal");
        let model = SemanticModel::new(&db, file);
        let (annotation, model) = model
            .enter_string_annotation(annotation)
            .expect("annotation should parse as a string annotation");
        let names = loaded_names_in_expression(annotation.expr(), &["first", "second"]);
        let definitions = definition_texts_for_names(&db, &module, source, &model, names);

        assert_eq!(definitions, ["first", "second"]);
    }

    fn definition_texts_for_names<'ast>(
        db: &'ast dyn crate::Db,
        module: &ParsedModuleRef,
        source: &str,
        model: &SemanticModel<'ast>,
        names: Vec<&'ast ast::ExprName>,
    ) -> Vec<String> {
        let mut definitions = Vec::new();
        for name in names {
            let load = model.reaching_definitions(name);
            assert!(
                load.is_some(),
                "inference should observe the requested name load at {:?}",
                name.range()
            );
            let load = load.expect("asserted that inference observed this name load");
            definitions.extend(load.definitions().iter().map(|definition| {
                let range = definition.full_range(db, module).range();
                source[range].to_string()
            }));
        }
        definitions
    }

    fn loaded_names_in_expression<'ast>(
        expression: &'ast ast::Expr,
        searched: &[&str],
    ) -> Vec<&'ast ast::ExprName> {
        let mut collector = NameCollector {
            searched,
            names: Vec::new(),
        };
        collector.visit_expr(expression);
        collector.names
    }

    struct NameCollector<'ast, 'name> {
        searched: &'name [&'name str],
        names: Vec<&'ast ast::ExprName>,
    }

    impl<'ast> Visitor<'ast> for NameCollector<'ast, '_> {
        fn visit_expr(&mut self, expression: &'ast ast::Expr) {
            if let ast::Expr::Name(name) = expression
                && name.ctx.is_load()
                && self.searched.contains(&name.id.as_str())
            {
                self.names.push(name);
            }
            walk_expr(self, expression);
        }
    }
}
