//! Computes source edits for Python module file renames.
//!
//! [`will_rename_files`] maps filesystem renames to module names, then rewrites imports and uses in
//! the candidate files supplied by the caller. It does not move files, discover package contents,
//! validate the filesystem operation, or normalize the returned edits.
//!
//! EXAMPLE:
//!
//! When `pkg/old.py` is renamed to `pkg/new.py`, the following code:
//!
//! ```python
//! from pkg import old
//! print(old.C)
//! ```
//!
//! is updated like so:
//!
//! ```python
//! from pkg import new
//! print(new.C)
//! ```
//!
//! [`will_rename_files`] plans edits in two AST passes per candidate file: one pass to compute
//! edits for imports, and then another to compute edits for references. [`edits_for_file`]
//! describes these passes. Beyond that, see [`module_rename`] for supported file renames,
//! [`ImportAnalyzer::import`] and [`ImportAnalyzer::import_from`] for import policies, and
//! [`ReferenceEditPlanner`] for reference policies.

use crate::RangedValue;
use rayon::prelude::*;
use ruff_db::files::{File, FileRange};
use ruff_db::source::source_text;
use ruff_db::system::{SystemPath, SystemPathBuf};
use ruff_python_ast::visitor::source_order::{SourceOrderVisitor, TraversalSignal};
use ruff_python_ast::{self as ast, AnyNodeRef};
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::{FxHashMap, FxHashSet};
use ty_module_resolver::{
    Module, ModuleName, ModuleResolveMode, ResolverEnvironment, ResolverFile, file_to_module,
    resolve_module_confident, resolve_real_module_confident, search_paths,
};
use ty_project::{Db, parallel::ParallelIteratorExt};
use ty_python_core::ProgramFile;
use ty_python_core::definition::{Definition, DefinitionKind};
use ty_python_semantic::types::Type;
use ty_python_semantic::{DefinitionResolution, HasType, SemanticModel};

/// Computes source edits for a batch of Python file renames.
///
/// `files` must include every source the caller wants analyzed, and the caller must restrict
/// `renames` and `files` to the intended scope. `db` must have place-load recording enabled.
///
/// Edits refer to the original files and source ranges. Unsupported or ambiguous occurrences are
/// omitted, so a non-empty result does not imply that every reference was updated. The caller must
/// sort the edits and handle duplicates and overlaps before applying them.
pub fn will_rename_files(
    db: &dyn Db,
    renames: &[FileRename],
    files: impl IntoIterator<Item = File>,
) -> Vec<FileRenameEdit> {
    let plan = RenamePlan::new(db, renames);
    if plan.module_renames.is_empty() {
        return Vec::new();
    }

    let mut files: Vec<_> = files.into_iter().collect();
    files.sort_unstable_by_key(|file| file.path(db).as_ref());
    files.dedup();

    files
        .into_par_iter()
        .map_with_db(db, |db, file| edits_for_file(db, file, &plan))
        .flatten()
        .collect()
}

/// One Python file rename in a batch.
pub struct FileRename {
    /// The source file before the rename.
    pub file: File,
    /// The destination path, which need not exist yet.
    pub new_path: SystemPathBuf,
}

/// A replacement and the file range containing it.
pub type FileRenameEdit = RangedValue<String>;

struct RenamePlan {
    // A map from old module name to new module name for supported rename operations.
    module_renames: FxHashMap<ModuleName, ModuleName>,
    // The last module name component for each key in `module_renames`.
    // This is used to filter out irrelevant files and identifiers to avoid unnecessary semantic analysis.
    old_module_basenames: FxHashSet<String>,
}

impl RenamePlan {
    fn new(db: &dyn Db, renames: &[FileRename]) -> Self {
        let module_renames: FxHashMap<_, _> = renames
            .iter()
            .filter_map(|rename| module_rename(db, rename))
            .filter(|(old, new)| old != new)
            .collect();
        let old_module_basenames = module_renames
            .keys()
            .map(|name| name.last_component().to_owned())
            .collect();
        Self {
            module_renames,
            old_module_basenames,
        }
    }

    fn new_module_name(&self, old_module_name: &ModuleName) -> Option<&ModuleName> {
        self.module_renames.get(old_module_name)
    }
}

/// Maps a supported file rename to its old and new module names.
///
/// A `.py` or `.pyi` file must keep its extension: `old.py` cannot become `old.pyi`.
/// Directory renames and renames involving `__init__.py` or `__init__.pyi` are unsupported.
/// The module must keep its parent: `a.old` can become `a.new`, but not `b.new` or `new`.
///
/// When a module has both a runtime file and a stub, renaming `old.py` to `new.py` updates
/// `import old` to `import new`, whether or not `old.pyi` is renamed with it. Renaming only
/// the stub leaves imports unchanged because the runtime module is still named `old`.
/// Renaming both files produces one set of edits.
fn module_rename(db: &dyn Db, rename: &FileRename) -> Option<(ModuleName, ModuleName)> {
    let resolver_environment = resolver_environment(db);
    let file = rename.file;
    let old = file.path(db).as_system_path()?;
    let new = SystemPath::absolute(&rename.new_path, db.system().current_directory());
    let extension = old.extension()?;

    if !matches!(extension, "py" | "pyi")
        || new.extension() != Some(extension)
        || old.file_stem() == Some("__init__")
        || new.file_stem() == Some("__init__")
    {
        return None;
    }

    let old_name = file_to_module(db, ResolverFile::new(db, file, resolver_environment))?
        .name(db)
        .clone();

    // A runtime module and its stub share an import name. Renaming only the stub
    // must not redirect imports while the runtime module remains at its old path.
    if resolve_module_file(db, &old_name)? != file {
        return None;
    }

    let new_name = destination_module_name(db, &new)?;
    if old_name.parent() != new_name.parent() {
        return None;
    }

    Some((old_name, new_name))
}

/// Derives a destination module name without requiring the destination to exist yet.
fn destination_module_name(db: &dyn Db, path: &SystemPath) -> Option<ModuleName> {
    search_paths(db, resolver_environment(db), ModuleResolveMode::Typing)
        .filter(|search_path| !search_path.is_standard_library())
        .find_map(|search_path| search_path.module_name_for_system_path(path))
}

fn resolve_module_file(db: &dyn Db, name: &ModuleName) -> Option<File> {
    let resolver_environment = resolver_environment(db);
    resolve_real_module_confident(db, resolver_environment, name)
        .or_else(|| resolve_module_confident(db, resolver_environment, name))?
        .file(db)
}

fn resolver_environment(db: &dyn Db) -> ResolverEnvironment<'_> {
    db.project().program(db).resolver_environment(db)
}

/// Plans edits for one candidate file in two AST passes.
///
/// 1. [`ImportEditPlanner`] plans supported import edits and records which local bindings change.
/// 2. [`ReferenceEditPlanner`] plans edits to module attributes and name reads, including those in
///    string annotations. Reaching definitions determine which name reads need updating when an
///    import changes a local binding.
///
/// Planning imports first lets us base reference edits on the actual binding changes, including
/// aliases and re-exports. A rejected import edit leaves its local binding unchanged.
fn edits_for_file(db: &dyn Db, file: File, plan: &RenamePlan) -> Vec<FileRenameEdit> {
    let program_file = db.program_file(file);
    let source = source_text(db, file);
    if source.read_error().is_some() {
        return Vec::new();
    }

    // Skip files whose source text contains none of the old module basenames.
    //
    // Only apply this check when both the source and the basenames are ASCII,
    // because non-ASCII identifiers can normalize to a different spelling in the AST.
    if source.as_str().is_ascii()
        && plan.old_module_basenames.iter().all(|name| name.is_ascii())
        && !plan
            .old_module_basenames
            .iter()
            .any(|name| source.as_str().contains(name))
    {
        return Vec::new();
    }

    let module = ruff_db::parsed::parsed_module(db, program_file.python_file(db)).load(db);
    let root = AnyNodeRef::from(module.syntax());
    let model = SemanticModel::new(db, program_file);

    // Determine which imports can be rewritten and which local bindings those edits rename.
    // Later reference edits depend on the binding rewrites recorded here.
    let mut imports = ImportEditPlanner::new(db, &model, plan);
    root.visit_source_order(&mut imports);

    // Plan reference edits, including those in string annotations.
    let mut references =
        ReferenceEditPlanner::new(db, &model, plan, &imports.output.definition_rewrites);
    root.visit_source_order(&mut references);

    let mut edits = imports.output.edits;
    edits.extend(references.edits);
    edits
}

type DefinitionRewrites<'db> = FxHashMap<Definition<'db>, String>;

#[derive(Default)]
struct ImportAnalysis<'db> {
    /// Source ranges and replacement text for edits to import statements.
    edits: Vec<FileRenameEdit>,
    /// Maps each affected import definition to replacement text for its references.
    definition_rewrites: DefinitionRewrites<'db>,
}

impl ImportAnalysis<'_> {
    fn is_empty(&self) -> bool {
        self.edits.is_empty() && self.definition_rewrites.is_empty()
    }

    fn extend(&mut self, other: Self) {
        self.edits.extend(other.edits);
        self.definition_rewrites.extend(other.definition_rewrites);
    }
}

struct ImportEditPlanner<'a, 'db> {
    analyzer: ImportAnalyzer<'a, 'db>,
    output: ImportAnalysis<'db>,
}

impl<'a, 'db> ImportEditPlanner<'a, 'db> {
    fn new(db: &'db dyn Db, model: &'a SemanticModel<'db>, plan: &'a RenamePlan) -> Self {
        Self {
            analyzer: ImportAnalyzer::new(db, model, plan),
            output: ImportAnalysis::default(),
        }
    }
}

impl<'a> SourceOrderVisitor<'a> for ImportEditPlanner<'a, '_> {
    fn enter_node(&mut self, node: AnyNodeRef<'a>) -> TraversalSignal {
        let output = match node {
            AnyNodeRef::StmtImport(import) => self.analyzer.import(import),
            AnyNodeRef::StmtImportFrom(import) => self
                .analyzer
                .import_from(import, &mut ExportAnalyzer::default()),
            _ => return TraversalSignal::Traverse,
        };

        if let Some(output) = output {
            self.output.extend(output);
        }

        TraversalSignal::Skip
    }

    // Import statements cannot occur inside expressions.
    fn visit_expr(&mut self, _expr: &'a ast::Expr) {}
}

/// Plans import edits and records the local bindings they change.
///
/// Statements keep their `import` or `from ... import` form.
struct ImportAnalyzer<'a, 'db> {
    db: &'db dyn Db,
    model: &'a SemanticModel<'db>,
    plan: &'a RenamePlan,
}

impl<'a, 'db> ImportAnalyzer<'a, 'db> {
    fn new(db: &'db dyn Db, model: &'a SemanticModel<'db>, plan: &'a RenamePlan) -> Self {
        Self { db, model, plan }
    }

    /// Plans each name in an `import` statement independently.
    /// Returns `None` if no edits or binding rewrites are produced.
    fn import(&self, import: &ast::StmtImport) -> Option<ImportAnalysis<'db>> {
        let mut output = ImportAnalysis::default();

        for alias in &import.names {
            let Some(old_module) = self.model.resolve_module(Some(alias.name.as_str()), 0) else {
                continue;
            };
            let Some(new_name) = self.plan.new_module_name(old_module.name(self.db)) else {
                continue;
            };
            let old_name = old_module.name(self.db);

            if let Some((definition, replacement)) = self.definition_rewrite(
                alias,
                old_name.first_component(),
                new_name.first_component(),
            ) {
                output.definition_rewrites.insert(definition, replacement);
            }

            if alias.name.as_str() != new_name.as_str() {
                output.edits.push(RangedValue {
                    range: FileRange::new(self.model.file(), alias.name.range),
                    value: new_name.as_str().to_string(),
                });
            }
        }

        (!output.is_empty()).then_some(output)
    }

    /// Plans edits to a `from` import and the local bindings it changes.
    ///
    /// Absolute and relative imports keep their prefixes: renaming `a.old` to `a.new`
    /// changes `from a.old import C` to `from a.new import C` and `from .old import C`
    /// to `from .new import C`.
    fn import_from(
        &self,
        import: &ast::StmtImportFrom,
        exports: &mut ExportAnalyzer<'db>,
    ) -> Option<ImportAnalysis<'db>> {
        let mut output = ImportAnalysis::default();

        let old_parent_module = self.model.resolve_module(
            import.module.as_ref().map(ast::Identifier::as_str),
            import.level,
        )?;
        let old_parent_name = old_parent_module.name(self.db);
        let new_parent_name = self.plan.new_module_name(old_parent_name);

        // Leave the statement unchanged if a name cannot be resolved in the renamed module.
        if new_parent_name.is_some()
            && import.names.iter().any(|alias| {
                alias
                    .inferred_type(self.model)
                    .is_none_or(|ty| ty.is_unknown())
            })
        {
            return None;
        }

        for alias in &import.names {
            self.import_from_alias(alias, old_parent_module, exports, &mut output)?;
        }

        if let Some(new_parent_name) = new_parent_name {
            let module_id = import.module.as_ref()?;

            // Only the last component changes; the written prefix and leading dots stay intact.
            let replacement = if let Some((prefix, _)) = module_id.as_str().rsplit_once('.') {
                format!("{prefix}.{}", new_parent_name.last_component())
            } else {
                new_parent_name.last_component().to_string()
            };

            output.edits.push(RangedValue {
                range: FileRange::new(self.model.file(), module_id.range),
                value: replacement,
            });
        }

        Some(output)
    }

    fn import_from_alias(
        &self,
        alias: &ast::Alias,
        old_parent_module: Module<'db>,
        exports: &mut ExportAnalyzer<'db>,
        output: &mut ImportAnalysis<'db>,
    ) -> Option<()> {
        let new_binding =
            if let Some(old_module) = self.directly_imported_submodule(alias, old_parent_module) {
                let Some(new_name) = self.plan.new_module_name(old_module.name(self.db)) else {
                    return Some(());
                };
                new_name.last_component().to_string()
            } else {
                if !self.plan.old_module_basenames.contains(alias.name.as_str()) {
                    return Some(());
                }
                match exports.module(
                    self.db,
                    self.model,
                    self.plan,
                    old_parent_module,
                    alias.name.as_str(),
                ) {
                    RewriteDecision::Preserve => return Some(()),
                    RewriteDecision::Replace(new_binding) => new_binding,
                    RewriteDecision::Omit => return None,
                }
            };

        output.definition_rewrites.extend(self.definition_rewrite(
            alias,
            alias.name.as_str(),
            &new_binding,
        ));
        output.edits.push(RangedValue {
            range: FileRange::new(self.model.file(), alias.name.range),
            value: new_binding,
        });
        Some(())
    }

    /// Identifies a submodule import such as `from pkg import old`, rather than a re-export.
    fn directly_imported_submodule(
        &self,
        alias: &ast::Alias,
        old_parent_module: Module<'db>,
    ) -> Option<Module<'db>> {
        let old_parent_name = old_parent_module.name(self.db);
        let imported_module = module_from_type(self.model, alias)?;
        let imported_name = imported_module.name(self.db);
        if alias.name.as_str() != imported_name.last_component()
            || imported_name.parent().as_ref() != Some(old_parent_name)
        {
            return None;
        }

        // Inside the package itself, this statement defines the export we are resolving.
        // Elsewhere, a package-level definition may preserve an explicit alias such as `old as old`.
        (self
            .model
            .definitions_for_module_global(old_parent_module, alias.name.as_str())
            .is_none()
            || file_to_module(self.db, self.model.program_file().resolver_file(self.db))
                .is_some_and(|importing_module| importing_module.name(self.db) == old_parent_name))
        .then_some(imported_module)
    }

    fn definition_rewrite(
        &self,
        alias: &ast::Alias,
        old: &str,
        new: &str,
    ) -> Option<(Definition<'db>, String)> {
        if alias.asname.is_some() || old == new {
            return None;
        }

        let definition = ty_python_core::semantic_index(self.db, self.model.program_file())
            .expect_single_definition(alias);
        Some((definition, new.to_string()))
    }
}

/// Shares complete import plans while following re-exports, rejecting cycles.
#[derive(Default)]
struct ExportAnalyzer<'db> {
    // `None` marks an import currently being visited.
    imports: FxHashMap<(ProgramFile<'db>, TextRange), Option<DefinitionRewrites<'db>>>,
    has_cycle: bool,
}

impl<'db> ExportAnalyzer<'db> {
    fn module(
        &mut self,
        db: &'db dyn Db,
        model: &SemanticModel<'db>,
        plan: &RenamePlan,
        module: Module<'db>,
        name: &str,
    ) -> RewriteDecision {
        let mut analyze = |module| {
            model.definitions_for_module_global(module, name).map_or(
                RewriteDecision::Omit,
                |resolution| {
                    rewrite_for_resolution(&resolution, |definition| {
                        self.definition(db, plan, definition)
                    })
                },
            )
        };
        let decision = analyze(module);
        if decision == RewriteDecision::Omit {
            return decision;
        }
        // A stub can expose a different binding. Propagate only changes both facets agree on.
        if let Some(runtime) =
            resolve_real_module_confident(db, resolver_environment(db), module.name(db))
            && runtime.file(db) != module.file(db)
            && analyze(runtime) != decision
        {
            return RewriteDecision::Omit;
        }
        decision
    }

    fn definition(
        &mut self,
        db: &'db dyn Db,
        plan: &RenamePlan,
        definition: Definition<'db>,
    ) -> RewriteDecision {
        let parsed = ruff_db::parsed::parsed_module(db, definition.python_file(db)).load(db);
        let model = SemanticModel::new(db, definition.program_file(db));
        match definition.kind(db) {
            DefinitionKind::Import(import) => {
                let import = import.import(&parsed);
                self.import(db, definition, import.range(), |_| {
                    ImportAnalyzer::new(db, &model, plan).import(import)
                })
            }
            DefinitionKind::ImportFrom(import) => {
                let import = import.import(&parsed);
                self.import(db, definition, import.range(), |exports| {
                    ImportAnalyzer::new(db, &model, plan).import_from(import, exports)
                })
            }
            DefinitionKind::StarImport(_) | DefinitionKind::ImportFromSubmodule(_) => {
                RewriteDecision::Omit
            }
            _ => RewriteDecision::Preserve,
        }
    }

    fn import(
        &mut self,
        db: &'db dyn Db,
        definition: Definition<'db>,
        range: TextRange,
        analyze: impl FnOnce(&mut Self) -> Option<ImportAnalysis<'db>>,
    ) -> RewriteDecision {
        if self.has_cycle {
            return RewriteDecision::Omit;
        }
        let key = (definition.program_file(db), range);
        if let Some(rewrites) = self.imports.get(&key) {
            return match rewrites {
                Some(rewrites) => rewrites
                    .get(&definition)
                    .map_or(RewriteDecision::Preserve, |new| {
                        RewriteDecision::Replace(new.clone())
                    }),
                None => {
                    self.has_cycle = true;
                    RewriteDecision::Omit
                }
            };
        }
        self.imports.insert(key, None);
        // A rejected statement leaves all its bindings unchanged. Cache the complete binding
        // map so looking up another name from this import does not plan the statement again.
        let rewrites = analyze(self)
            .map(|analysis| analysis.definition_rewrites)
            .unwrap_or_default();
        // Reject the whole traversal: a cycle can prevent an import rewrite, but that does not
        // establish that its exported binding stays unchanged.
        if self.has_cycle {
            return RewriteDecision::Omit;
        }
        let decision = rewrites
            .get(&definition)
            .map_or(RewriteDecision::Preserve, |new| {
                RewriteDecision::Replace(new.clone())
            });
        self.imports.insert(key, Some(rewrites));
        decision
    }
}

/// Plans edits to module reads, including those in string annotations such as `value: 'old.C'`.
///
/// Assignment and deletion targets such as `old = 0` and `del old` keep their names.
/// Reads within a target can still change: renaming `pkg/old.py` to `pkg/new.py` changes
/// `pkg.old.VALUE = 1` to `pkg.new.VALUE = 1`.
///
/// Declarations such as `global old` and `nonlocal old`, and name reads that depend on
/// them, are also left unchanged. Supporting these cases requires coordinating the binding's
/// declarations, writes, deletions, and reads.
///
/// Uses introduced by star imports such as `from pkg import *`, names in `__all__ = ["old"]`,
/// and dynamic references such as `importlib.import_module("pkg.old")` are unsupported.
/// See [`rewrite_for_resolution`] for how ambiguous references are handled.
struct ReferenceEditPlanner<'a, 'db> {
    db: &'db dyn Db,
    model: &'a SemanticModel<'db>,
    plan: &'a RenamePlan,
    definition_rewrites: &'a DefinitionRewrites<'db>,
    edits: Vec<FileRenameEdit>,
}

impl<'a, 'db> ReferenceEditPlanner<'a, 'db> {
    fn new(
        db: &'db dyn Db,
        model: &'a SemanticModel<'db>,
        plan: &'a RenamePlan,
        definition_rewrites: &'a DefinitionRewrites<'db>,
    ) -> Self {
        Self {
            db,
            model,
            plan,
            definition_rewrites,
            edits: Vec::new(),
        }
    }

    fn name(&mut self, name: &ast::ExprName) {
        let Some(resolution) = self.model.reaching_definitions(name) else {
            return;
        };
        if resolution.crosses_scope_declaration() {
            // Keep reads tied to the `global` or `nonlocal` declaration we leave unchanged.
            return;
        }
        let decision = rewrite_for_resolution(&resolution, |definition| {
            if matches!(definition.kind(self.db), DefinitionKind::StarImport(_)) {
                RewriteDecision::Omit
            } else if let Some(replacement) = self.definition_rewrites.get(&definition) {
                RewriteDecision::Replace(replacement.clone())
            } else {
                RewriteDecision::Preserve
            }
        });
        self.apply(name.range(), decision);
    }

    fn attribute(&mut self, attribute: &ast::ExprAttribute) {
        if !attribute.ctx.is_load() {
            // A write target's receiver can still contain module reads.
            return;
        }
        let Some(module) = module_from_type(self.model, attribute) else {
            return;
        };
        let Some(new) = self.plan.new_module_name(module.name(self.db)) else {
            return;
        };
        let Some(receiver) = module_from_type(self.model, &*attribute.value) else {
            return;
        };
        let decision = if self
            .model
            .definitions_for_module_global(receiver, attribute.attr.as_str())
            .is_some()
        {
            ExportAnalyzer::default().module(
                self.db,
                self.model,
                self.plan,
                receiver,
                attribute.attr.as_str(),
            )
        } else {
            replace(attribute.attr.as_str(), new.last_component())
        };
        self.apply(attribute.attr.range, decision);
    }

    fn string(&mut self, string: &ast::ExprStringLiteral) {
        let Some((ast, model)) = self.model.enter_string_annotation(string) else {
            return;
        };
        let mut planner =
            ReferenceEditPlanner::new(self.db, &model, self.plan, self.definition_rewrites);
        planner.visit_expr(ast.expr());
        self.edits.extend(planner.edits);
    }

    fn apply(&mut self, range: TextRange, decision: RewriteDecision) {
        if let RewriteDecision::Replace(text) = decision {
            self.edits.push(RangedValue {
                range: FileRange::new(self.model.file(), range),
                value: text,
            });
        }
    }
}

impl<'a> SourceOrderVisitor<'a> for ReferenceEditPlanner<'a, '_> {
    fn enter_node(&mut self, node: AnyNodeRef<'a>) -> TraversalSignal {
        match node {
            AnyNodeRef::ExprName(name)
                if name.ctx.is_load()
                    && self.plan.old_module_basenames.contains(name.id.as_str()) =>
            {
                self.name(name);
            }
            AnyNodeRef::ExprAttribute(attribute)
                if self
                    .plan
                    .old_module_basenames
                    .contains(attribute.attr.as_str()) =>
            {
                self.attribute(attribute);
            }
            AnyNodeRef::ExprStringLiteral(string) => {
                self.string(string);
                return TraversalSignal::Skip;
            }
            _ => {}
        }
        TraversalSignal::Traverse
    }
}

/// Chooses a rewrite only when all reachable definitions agree on the replacement or preservation.
///
/// Incomplete resolution or a reachable deletion prevents a rewrite. Possible unboundness alone
/// does not: a conditional import can still establish the same replacement wherever it is bound.
///
/// If branches execute `from a import x` and `from b import x`, renaming those modules to
/// `a.y` and `b.z` updates both imports but leaves a subsequent `print(x)` unchanged:
/// neither `y` nor `z` works for both branches. Such partial results can require manual fixes.
fn rewrite_for_resolution<'db>(
    resolution: &DefinitionResolution<'db>,
    mut rewrite_for: impl FnMut(Definition<'db>) -> RewriteDecision,
) -> RewriteDecision {
    if !resolution.is_complete() || resolution.may_be_deleted() {
        return RewriteDecision::Omit;
    }
    let Some((first, definitions)) = resolution.definitions().split_first() else {
        return RewriteDecision::Omit;
    };
    let rewrite = rewrite_for(*first);
    if rewrite == RewriteDecision::Omit {
        return rewrite;
    }
    if definitions
        .iter()
        .copied()
        .any(|definition| rewrite_for(definition) != rewrite)
    {
        return RewriteDecision::Omit;
    }
    rewrite
}

#[derive(Eq, PartialEq)]
enum RewriteDecision {
    /// The definition keeps its current spelling, for example because it has an explicit alias.
    Preserve,
    /// The definition requires this replacement wherever the name refers to it.
    Replace(String),
    /// Resolution cannot establish a rewrite, so this occurrence is left unchanged.
    Omit,
}

fn replace(old: &str, new: &str) -> RewriteDecision {
    if old != new {
        return RewriteDecision::Replace(new.to_string());
    }
    RewriteDecision::Preserve
}

fn module_from_type<'db, T: HasType>(
    model: &SemanticModel<'db>,
    expression: &T,
) -> Option<Module<'db>> {
    let Type::ModuleLiteral(literal) = expression.inferred_type(model)? else {
        return None;
    };
    Some(literal.module(model.db()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruff_db::files::system_path_to_file;
    use ruff_db::system::DbWithWritableSystem;
    use ruff_python_ast::PythonVersion;
    use ruff_text_size::Ranged;
    use std::collections::BTreeSet;
    use ty_project::{ProjectMetadata, TestDb};
    use ty_python_semantic::PlaceLoadRecordingMode;

    #[test]
    fn file_rename_contract() {
        let db = test_db(&[
            ("/pkg/__init__.py", ""),
            ("/pkg/old.py", "class C: ..."),
            (
                "/pkg/use.py",
                "from .old import C
from . import old
print(C, old.C)",
            ),
            (
                "/use.py",
                "import pkg.old
from pkg.old import C
from pkg import old
import pkg.old as stable
value: 'old.C'
runtime = 'old.C'
print(old, stable)
pkg.old.VALUE = 1
pkg.old = pkg.old
",
            ),
        ]);
        assert_success(
            &db,
            &[("/pkg/old.py", "/pkg/new.py")],
            &[
                (
                    "/pkg/use.py",
                    "from .new import C
from . import new
print(C, new.C)",
                ),
                (
                    "/use.py",
                    "import pkg.new
from pkg.new import C
from pkg import new
import pkg.new as stable
value: 'new.C'
runtime = 'old.C'
print(new, stable)
pkg.new.VALUE = 1
pkg.old = pkg.new
",
                ),
            ],
        );
    }

    #[test]
    fn unicode_identifier_prefilter() {
        let db = test_db(&[
            ("/K·b.py", ""),
            (
                "/use.py",
                "import \u{212a}·b
print(\u{212a}·b)
",
            ),
        ]);
        assert_success(
            &db,
            &[("/K·b.py", "/new.py")],
            &[(
                "/use.py",
                "import new
print(new)
",
            )],
        );
    }

    #[test]
    fn explicit_reexports_propagate_transitively() {
        let db = test_db(&[
            ("/pkg/__init__.py", ""),
            ("/pkg/old.py", ""),
            ("/facade.py", "from pkg import old"),
            ("/bridge.py", "from facade import old"),
            ("/stable.py", "from pkg import old as old"),
            (
                "/use.py",
                "from bridge import old
import bridge, stable
print(old, bridge.old, stable.old)",
            ),
            (
                "/star.py",
                "from bridge import *
old",
            ),
        ]);
        assert_success(
            &db,
            &[("/pkg/old.py", "/pkg/new.py")],
            &[
                ("/facade.py", "from pkg import new"),
                ("/bridge.py", "from facade import new"),
                ("/stable.py", "from pkg import new as old"),
                (
                    "/use.py",
                    "from bridge import new
import bridge, stable
print(new, bridge.new, stable.old)",
                ),
            ],
        );
    }

    #[test]
    fn runtime_and_stub_module_facets() {
        let db = test_db(&[
            ("/pkg/__init__.py", ""),
            ("/pkg/old.py", ""),
            ("/pkg/old.pyi", ""),
            (
                "/use.py",
                "from pkg import old
print(old)
",
            ),
        ]);

        assert_file_no_edits(&db, "/pkg/old.pyi", "/pkg/new.pyi");
        assert_success(
            &db,
            &[("/pkg/old.py", "/pkg/new.py")],
            &[(
                "/use.py",
                "from pkg import new
print(new)
",
            )],
        );
        assert_success(
            &db,
            &[
                ("/pkg/old.py", "/pkg/new.py"),
                ("/pkg/old.pyi", "/pkg/new.pyi"),
            ],
            &[(
                "/use.py",
                "from pkg import new
print(new)
",
            )],
        );
    }

    #[test]
    fn conflicting_runtime_and_stub_aliases_are_omitted() {
        let consumer = "import pkg
print(pkg.old)
";
        let db = test_db(&[
            ("/pkg/__init__.py", "from . import old as old\n"),
            ("/pkg/__init__.pyi", "from . import old\n"),
            ("/pkg/old.py", ""),
            ("/use.py", consumer),
        ]);
        assert_success(
            &db,
            &[("/pkg/old.py", "/pkg/new.py")],
            &[
                ("/pkg/__init__.py", "from . import new as old\n"),
                ("/pkg/__init__.pyi", "from . import new\n"),
            ],
        );
    }

    #[test]
    fn possibly_unbound_place_loads_are_rewritten() {
        let db = test_db(&[
            ("/old.py", ""),
            (
                "/use.py",
                "def f(flag):
    if flag:
        import old
    return old",
            ),
        ]);
        assert_success(
            &db,
            &[("/old.py", "/new.py")],
            &[(
                "/use.py",
                "def f(flag):
    if flag:
        import new
    return new",
            )],
        );
    }

    #[test]
    fn scope_declarations_omit_only_dependent_occurrences() {
        let db = test_db(&[
            ("/old.py", ""),
            (
                "/use.py",
                "import old
def affected():
    global old
    return old
def sibling():
    return old",
            ),
        ]);
        assert_success(
            &db,
            &[("/old.py", "/new.py")],
            &[(
                "/use.py",
                "import new
def affected():
    global old
    return old
def sibling():
    return new",
            )],
        );
    }

    #[test]
    fn unsupported_requests_and_imports_are_omitted() {
        let mut db = test_db(&[
            ("/a/__init__.py", ""),
            ("/a/old.py", ""),
            ("/b/__init__.py", ""),
            ("/use.py", "from a import old"),
        ]);
        assert_file_no_edits(&db, "/a/old.py", "/b/new.py");
        db.write_file("/use.py", "import a.old.missing\n").unwrap();
        assert_file_no_edits(&db, "/a/old.py", "/a/new.py");
        assert_file_no_edits(&db, "/a/__init__.py", "/a/new.py");
    }

    #[test]
    fn conflicting_place_load_definitions_are_omitted() {
        let db = test_db(&[
            ("/a/__init__.py", ""),
            ("/a/x.py", ""),
            ("/b/__init__.py", ""),
            ("/b/x.py", ""),
            (
                "/use.py",
                "if flag: from a import x
else: from b import x
print(x)
",
            ),
        ]);
        assert_success(
            &db,
            &[("/a/x.py", "/a/y.py"), ("/b/x.py", "/b/z.py")],
            &[(
                "/use.py",
                "if flag: from a import y
else: from b import z
print(x)
",
            )],
        );
    }

    #[test]
    fn edits_only_candidate_files() {
        let db = test_db(&[
            ("/old.py", ""),
            ("/included.py", "import old\n"),
            ("/not_a_candidate.py", "import old\n"),
        ]);
        let included = system_path_to_file(&db, "/included.py").unwrap();
        let rename = file_renames(&db, &[("/old.py", "/new.py")]);

        let edits = will_rename_files(&db, &rename, [included]);
        assert!(edits.iter().all(|edit| edit.range.file() == included));
        assert_eq!(apply_edits(&db, &edits, "/included.py"), "import new\n");
    }

    #[test]
    fn unsupported_rules_do_not_suppress_independent_edits() {
        let db = test_db(&[
            ("/unsupported.py", ""),
            ("/old.py", ""),
            ("/use.py", "import old\n"),
        ]);
        assert_success(
            &db,
            &[
                ("/unsupported.py", "/unsupported.pyi"),
                ("/old.py", "/new.py"),
            ],
            &[("/use.py", "import new\n")],
        );
    }

    fn file_renames(db: &TestDb, paths: &[(&str, &str)]) -> Vec<FileRename> {
        paths
            .iter()
            .map(|&(old, new)| FileRename {
                file: system_path_to_file(db, old).unwrap(),
                new_path: new.into(),
            })
            .collect()
    }

    #[track_caller]
    fn assert_file_no_edits(db: &TestDb, old: &str, new: &str) {
        assert_no_edits(old, db, &[(old, new)]);
    }

    #[track_caller]
    fn assert_success(db: &TestDb, renames: &[(&str, &str)], expected: &[(&str, &str)]) {
        let renames = file_renames(db, renames);
        let edits = will_rename_files(db, &renames, &db.project().files(db));
        let actual: BTreeSet<_> = edits.iter().map(|edit| edit.range.file()).collect();
        let expected_files: BTreeSet<_> = expected
            .iter()
            .map(|(path, _)| system_path_to_file(db, *path).unwrap())
            .collect();
        assert_eq!(actual, expected_files);
        for &(path, contents) in expected {
            assert_eq!(apply_edits(db, &edits, path), contents, "{path}");
        }
    }

    #[track_caller]
    fn assert_no_edits(name: &str, db: &TestDb, renames: &[(&str, &str)]) {
        let renames = file_renames(db, renames);
        let edits = will_rename_files(db, &renames, &db.project().files(db));
        assert!(edits.is_empty(), "{name}: {edits:?}");
    }

    fn test_db(files: &[(&str, &str)]) -> TestDb {
        let mut db = TestDb::with_place_load_recording_mode(
            ProjectMetadata::new("test", "/".into()),
            PlaceLoadRecordingMode::Enabled,
        );
        db.set_python_version(PythonVersion::latest_ty());
        db.write_files(files.iter().copied()).unwrap();
        db
    }

    fn apply_edits(db: &TestDb, edits: &[FileRenameEdit], path: &str) -> String {
        let file = system_path_to_file(db, path).unwrap();
        let mut edits: Vec<_> = edits
            .iter()
            .filter(|edit| edit.range.file() == file)
            .collect();
        edits.sort_unstable_by_key(|edit| std::cmp::Reverse(edit.range.start()));
        let mut result = source_text(db, file).as_str().to_owned();
        for edit in edits {
            result.replace_range(edit.range.range().to_std_range(), &edit.value);
        }
        result
    }
}
