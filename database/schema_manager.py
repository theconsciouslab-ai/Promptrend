# database/schema_manager.py
import logging
import json
from datetime import datetime
import data_collection_config

logger = logging.getLogger("SchemaManager")

class SchemaManager:
    """
    Manages schema evolution for the vulnerability database.
    
    Implements a structured approach to schema changes that preserves
    backward compatibility and enables phased feature rollout.
    """
    
    def __init__(self, storage_manager):
        """
        Initialize the schema manager.
        
        Args:
            storage_manager: Storage manager for database operations
        """
        self.storage_manager = storage_manager
        
        # Current schema versions
        self.current_versions = data_collection_config.SCHEMA_VERSIONS
        
        # Load migration definitions
        self.migrations = self._load_migrations()
        
        # Track schema version for documents
        self.document_versions = {}
    
    def _load_migrations(self):
        """
        Load migration definitions.
        
        Returns:
            dict: Migration definitions
        """
        try:
            with open("config/migrations.json", "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading migrations: {str(e)}")
            return {}
    
    def check_and_migrate_document(self, document):
        """
        Check if a document needs migration and perform it if needed.
        
        Args:
            document: Document to check
            
        Returns:
            dict: Migrated document
        """
        # Get document schema version
        doc_version = document.get("_schema_version", "1.0")
        
        # Get current version
        current_version = self.current_versions.get("document", "1.0")
        
        # Check if migration needed
        if doc_version != current_version:
            logger.info(f"Migrating document {document.get('uuid')} from v{doc_version} to v{current_version}")
            
            # Perform migration
            migrated = self._migrate_document(document, doc_version, current_version)
            
            # Update version
            migrated["_schema_version"] = current_version
            
            # Update in storage
            self.storage_manager.update_document(migrated)
            
            return migrated
        
        return document
    
    def _migrate_document(self, document, from_version, to_version):
        """
        Migrate a document from one version to another.
        
        Args:
            document: Document to migrate
            from_version: Source version
            to_version: Target version
            
        Returns:
            dict: Migrated document
        """
        # Clone document to avoid modifying original
        migrated = {**document}
        
        # Find migration path
        migration_path = self._get_migration_path(from_version, to_version)
        
        if not migration_path:
            logger.warning(f"No migration path from v{from_version} to v{to_version}")
            return document
        
        # Apply migrations in sequence
        for step in migration_path:
            migrated = self._apply_migration_step(migrated, step)
        
        return migrated
    
    def _get_migration_path(self, from_version, to_version):
        """
        Determine the migration path between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            
        Returns:
            list: Migration steps
        """
        # Check for direct migration
        direct_key = f"{from_version}_to_{to_version}"
        if direct_key in self.migrations:
            return [direct_key]
        
        # Otherwise, find a path
        version_graph = self._build_version_graph()
        return self._find_shortest_path(version_graph, from_version, to_version)
    
    def _build_version_graph(self):
        """
        Build a graph of version relationships from migrations.
        
        Returns:
            dict: Version graph
        """
        graph = {}
        
        for migration_key in self.migrations.keys():
            if "_to_" in migration_key:
                from_version, to_version = migration_key.split("_to_")
                
                if from_version not in graph:
                    graph[from_version] = []
                
                graph[from_version].append(to_version)
        
        return graph
    
    def _find_shortest_path(self, graph, start, end):
        """
        Find shortest path in version graph using BFS.
        
        Args:
            graph: Version graph
            start: Start version
            end: End version
            
        Returns:
            list: Migration path or None if no path exists
        """
        if start == end:
            return []
            
        if start not in graph:
            return None
        
        # BFS to find shortest path
        queue = [(start, [])]
        visited = set()
        
        while queue:
            (vertex, path) = queue.pop(0)
            
            if vertex not in visited:
                # Check if we reached the target
                if vertex == end:
                    return [f"{path[i]}_to_{path[i+1]}" for i in range(len(path)-1)]
                
                # Mark as visited
                visited.add(vertex)
                
                # Enqueue neighbors
                for neighbor in graph.get(vertex, []):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def _apply_migration_step(self, document, migration_key):
        """
        Apply a single migration step to a document.
        
        Args:
            document: Document to migrate
            migration_key: Migration step key
            
        Returns:
            dict: Migrated document
        """
        migration = self.migrations.get(migration_key, {})
        migrated = {**document}
        
        # Apply field additions
        for field, value in migration.get("add_fields", {}).items():
            if field not in migrated:
                migrated[field] = value
        
        # Apply field removals
        for field in migration.get("remove_fields", []):
            if field in migrated:
                del migrated[field]
        
        # Apply field renames
        for old_field, new_field in migration.get("rename_fields", {}).items():
            if old_field in migrated:
                migrated[new_field] = migrated.pop(old_field)
        
        # Apply structure transformations
        for transform in migration.get("transforms", []):
            migrated = self._apply_transform(migrated, transform)
        
        # Add migration metadata
        if "_migration_history" not in migrated:
            migrated["_migration_history"] = []
            
        migrated["_migration_history"].append({
            "migration": migration_key,
            "applied_at": datetime.now().isoformat()
        })
        
        return migrated
    
    def _apply_transform(self, document, transform):
        """
        Apply a transformation to a document.
        
        Args:
            document: Document to transform
            transform: Transformation definition
            
        Returns:
            dict: Transformed document
        """
        # Clone document to avoid modifying original
        transformed = {**document}
        
        # Get transformation type
        transform_type = transform.get("type")
        
        if transform_type == "nested_structure":
            # Create nested structure
            target_field = transform.get("target_field")
            source_fields = transform.get("source_fields", [])
            
            # Create new nested structure
            nested = {}
            
            for field in source_fields:
                if field in transformed:
                    nested[field] = transformed.pop(field)
            
            # Add nested structure
            transformed[target_field] = nested
            
        elif transform_type == "flatten_structure":
            # Flatten nested structure
            source_field = transform.get("source_field")
            
            if source_field in transformed and isinstance(transformed[source_field], dict):
                # Extract nested fields
                nested = transformed.pop(source_field)
                
                # Add to root
                for field, value in nested.items():
                    transformed[field] = value
        
        elif transform_type == "convert_type":
            # Convert field type
            target_field = transform.get("target_field")
            target_type = transform.get("target_type")
            
            if target_field in transformed:
                # Convert based on target type
                if target_type == "string":
                    transformed[target_field] = str(transformed[target_field])
                elif target_type == "integer":
                    transformed[target_field] = int(float(transformed[target_field]))
                elif target_type == "float":
                    transformed[target_field] = float(transformed[target_field])
                elif target_type == "boolean":
                    transformed[target_field] = bool(transformed[target_field])
                elif target_type == "array":
                    if not isinstance(transformed[target_field], list):
                        transformed[target_field] = [transformed[target_field]]
        
        return transformed