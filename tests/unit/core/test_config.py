"""
Unit tests for configuration system.

Tests schema validation, configuration loading, and validation.
"""

import unittest
from unittest.mock import Mock, patch, mock_open
import json
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.config.schemas import (
    ConfigSchema,
    SchemaField,
    SchemaValidator,
    ValidationError
)
from src.core.config.simple_validator import SimpleValidator
from src.core.config.schema_validator import SchemaValidationEngine


class TestSchemaField(unittest.TestCase):
    """Test schema field definitions."""
    
    def test_field_creation(self):
        """Test creating schema fields."""
        field = SchemaField(
            name="port",
            type="integer",
            required=True,
            default=8080,
            min_value=1,
            max_value=65535,
            description="Server port number"
        )
        
        self.assertEqual(field.name, "port")
        self.assertEqual(field.type, "integer")
        self.assertTrue(field.required)
        self.assertEqual(field.default, 8080)
        self.assertEqual(field.min_value, 1)
        self.assertEqual(field.max_value, 65535)
    
    def test_field_validation(self):
        """Test field validation."""
        field = SchemaField(
            name="percentage",
            type="float",
            min_value=0.0,
            max_value=100.0
        )
        
        # Valid values
        self.assertTrue(field.validate(50.0))
        self.assertTrue(field.validate(0.0))
        self.assertTrue(field.validate(100.0))
        
        # Invalid values
        self.assertFalse(field.validate(-1.0))
        self.assertFalse(field.validate(101.0))
        self.assertFalse(field.validate("not a number"))
    
    def test_field_with_choices(self):
        """Test field with enumerated choices."""
        field = SchemaField(
            name="log_level",
            type="string",
            choices=["debug", "info", "warning", "error"],
            default="info"
        )
        
        self.assertTrue(field.validate("debug"))
        self.assertTrue(field.validate("error"))
        self.assertFalse(field.validate("trace"))
        self.assertFalse(field.validate("INFO"))  # Case sensitive
    
    def test_nested_field(self):
        """Test nested object field."""
        field = SchemaField(
            name="database",
            type="object",
            properties={
                "host": SchemaField("host", "string", required=True),
                "port": SchemaField("port", "integer", default=5432),
                "name": SchemaField("name", "string", required=True)
            }
        )
        
        # Valid nested object
        valid_config = {
            "host": "localhost",
            "port": 5432,
            "name": "test_db"
        }
        self.assertTrue(field.validate(valid_config))
        
        # Missing required field
        invalid_config = {
            "host": "localhost",
            "port": 5432
            # Missing 'name'
        }
        self.assertFalse(field.validate(invalid_config))


class TestConfigSchema(unittest.TestCase):
    """Test configuration schema."""
    
    def test_schema_creation(self):
        """Test creating configuration schema."""
        schema = ConfigSchema(
            name="AppConfig",
            version="1.0.0",
            fields=[
                SchemaField("host", "string", required=True),
                SchemaField("port", "integer", default=8080),
                SchemaField("debug", "boolean", default=False)
            ]
        )
        
        self.assertEqual(schema.name, "AppConfig")
        self.assertEqual(schema.version, "1.0.0")
        self.assertEqual(len(schema.fields), 3)
    
    def test_schema_validation(self):
        """Test validating config against schema."""
        schema = ConfigSchema(
            name="ServerConfig",
            fields=[
                SchemaField("host", "string", required=True),
                SchemaField("port", "integer", required=True, min_value=1, max_value=65535),
                SchemaField("workers", "integer", default=4, min_value=1)
            ]
        )
        
        # Valid config
        valid_config = {
            "host": "0.0.0.0",
            "port": 8080,
            "workers": 8
        }
        
        result = schema.validate(valid_config)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        
        # Invalid config - missing required field
        invalid_config = {
            "port": 8080
        }
        
        result = schema.validate(invalid_config)
        self.assertFalse(result.is_valid)
        self.assertIn("host", str(result.errors[0]))
        
        # Invalid config - value out of range
        invalid_config2 = {
            "host": "localhost",
            "port": 70000,  # Out of range
            "workers": 0     # Below minimum
        }
        
        result = schema.validate(invalid_config2)
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 2)
    
    def test_schema_with_defaults(self):
        """Test schema applying defaults."""
        schema = ConfigSchema(
            name="DefaultConfig",
            fields=[
                SchemaField("name", "string", required=True),
                SchemaField("timeout", "integer", default=30),
                SchemaField("retries", "integer", default=3),
                SchemaField("verbose", "boolean", default=False)
            ]
        )
        
        # Minimal config
        config = {"name": "test"}
        
        # Apply defaults
        enriched = schema.apply_defaults(config)
        
        self.assertEqual(enriched["name"], "test")
        self.assertEqual(enriched["timeout"], 30)
        self.assertEqual(enriched["retries"], 3)
        self.assertEqual(enriched["verbose"], False)
    
    def test_nested_schema(self):
        """Test nested schema validation."""
        schema = ConfigSchema(
            name="AppConfig",
            fields=[
                SchemaField("app_name", "string", required=True),
                SchemaField(
                    "database",
                    "object",
                    required=True,
                    properties={
                        "host": SchemaField("host", "string", required=True),
                        "port": SchemaField("port", "integer", default=5432),
                        "credentials": SchemaField(
                            "credentials",
                            "object",
                            properties={
                                "username": SchemaField("username", "string", required=True),
                                "password": SchemaField("password", "string", required=True)
                            }
                        )
                    }
                ),
                SchemaField(
                    "features",
                    "array",
                    item_type="string",
                    default=[]
                )
            ]
        )
        
        # Valid nested config
        config = {
            "app_name": "TestApp",
            "database": {
                "host": "db.example.com",
                "credentials": {
                    "username": "admin",
                    "password": "secret"
                }
            },
            "features": ["feature1", "feature2"]
        }
        
        result = schema.validate(config)
        self.assertTrue(result.is_valid)
        
        # Apply defaults to nested
        enriched = schema.apply_defaults(config)
        self.assertEqual(enriched["database"]["port"], 5432)


class TestSimpleValidator(unittest.TestCase):
    """Test simple validator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = SimpleValidator()
    
    def test_type_validation(self):
        """Test basic type validation."""
        # String validation
        self.assertTrue(self.validator.validate_type("hello", "string"))
        self.assertFalse(self.validator.validate_type(123, "string"))
        
        # Integer validation
        self.assertTrue(self.validator.validate_type(42, "integer"))
        self.assertFalse(self.validator.validate_type(42.5, "integer"))
        self.assertFalse(self.validator.validate_type("42", "integer"))
        
        # Float validation
        self.assertTrue(self.validator.validate_type(3.14, "float"))
        self.assertTrue(self.validator.validate_type(3, "float"))  # Int is valid float
        self.assertFalse(self.validator.validate_type("3.14", "float"))
        
        # Boolean validation
        self.assertTrue(self.validator.validate_type(True, "boolean"))
        self.assertTrue(self.validator.validate_type(False, "boolean"))
        self.assertFalse(self.validator.validate_type(1, "boolean"))
        
        # Array validation
        self.assertTrue(self.validator.validate_type([1, 2, 3], "array"))
        self.assertTrue(self.validator.validate_type([], "array"))
        self.assertFalse(self.validator.validate_type((1, 2, 3), "array"))
        
        # Object validation
        self.assertTrue(self.validator.validate_type({"key": "value"}, "object"))
        self.assertTrue(self.validator.validate_type({}, "object"))
        self.assertFalse(self.validator.validate_type([], "object"))
    
    def test_constraint_validation(self):
        """Test constraint validation."""
        # Min/max constraints
        constraints = {"min_value": 0, "max_value": 100}
        self.assertTrue(self.validator.validate_constraints(50, constraints))
        self.assertTrue(self.validator.validate_constraints(0, constraints))
        self.assertTrue(self.validator.validate_constraints(100, constraints))
        self.assertFalse(self.validator.validate_constraints(-1, constraints))
        self.assertFalse(self.validator.validate_constraints(101, constraints))
        
        # Length constraints
        constraints = {"min_length": 3, "max_length": 10}
        self.assertTrue(self.validator.validate_constraints("hello", constraints))
        self.assertFalse(self.validator.validate_constraints("hi", constraints))
        self.assertFalse(self.validator.validate_constraints("hello world!", constraints))
        
        # Pattern constraint
        constraints = {"pattern": r"^[a-z]+$"}
        self.assertTrue(self.validator.validate_constraints("hello", constraints))
        self.assertFalse(self.validator.validate_constraints("Hello", constraints))
        self.assertFalse(self.validator.validate_constraints("hello123", constraints))
    
    def test_custom_validators(self):
        """Test custom validation functions."""
        def is_even(value):
            return value % 2 == 0
        
        def is_uppercase(value):
            return value.isupper()
        
        self.validator.add_custom_validator("even", is_even)
        self.validator.add_custom_validator("uppercase", is_uppercase)
        
        # Test custom validators
        self.assertTrue(self.validator.validate_custom(4, "even"))
        self.assertFalse(self.validator.validate_custom(3, "even"))
        
        self.assertTrue(self.validator.validate_custom("HELLO", "uppercase"))
        self.assertFalse(self.validator.validate_custom("Hello", "uppercase"))


class TestSchemaValidationEngine(unittest.TestCase):
    """Test schema validation engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = SchemaValidationEngine()
    
    def test_register_schema(self):
        """Test registering schemas."""
        schema = ConfigSchema(
            name="TestSchema",
            version="1.0.0",
            fields=[
                SchemaField("field1", "string", required=True),
                SchemaField("field2", "integer", default=42)
            ]
        )
        
        self.engine.register_schema("test", schema)
        
        # Should be able to retrieve
        retrieved = self.engine.get_schema("test")
        self.assertEqual(retrieved.name, "TestSchema")
    
    def test_validate_with_schema(self):
        """Test validating config with registered schema."""
        schema = ConfigSchema(
            name="AppSchema",
            fields=[
                SchemaField("name", "string", required=True),
                SchemaField("port", "integer", required=True)
            ]
        )
        
        self.engine.register_schema("app", schema)
        
        # Valid config
        config = {"name": "MyApp", "port": 8080}
        result = self.engine.validate("app", config)
        self.assertTrue(result.is_valid)
        
        # Invalid config
        config = {"name": "MyApp"}  # Missing port
        result = self.engine.validate("app", config)
        self.assertFalse(result.is_valid)
    
    def test_load_schema_from_file(self):
        """Test loading schema from file."""
        schema_json = {
            "name": "FileSchema",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "setting1",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "setting2",
                    "type": "integer",
                    "default": 100
                }
            ]
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(schema_json))):
            self.engine.load_schema_from_file("schema.json", "file_schema")
        
        # Should be registered
        schema = self.engine.get_schema("file_schema")
        self.assertEqual(schema.name, "FileSchema")
        self.assertEqual(len(schema.fields), 2)
    
    def test_validation_with_inheritance(self):
        """Test schema inheritance."""
        # Base schema
        base_schema = ConfigSchema(
            name="BaseConfig",
            fields=[
                SchemaField("id", "string", required=True),
                SchemaField("created_at", "string", required=True)
            ]
        )
        
        # Extended schema
        extended_schema = ConfigSchema(
            name="ExtendedConfig",
            parent=base_schema,
            fields=[
                SchemaField("name", "string", required=True),
                SchemaField("value", "integer", default=0)
            ]
        )
        
        self.engine.register_schema("extended", extended_schema)
        
        # Should validate fields from both schemas
        config = {
            "id": "123",
            "created_at": "2024-01-01",
            "name": "Test",
            "value": 42
        }
        
        result = self.engine.validate("extended", config)
        self.assertTrue(result.is_valid)
        
        # Missing base field
        config = {
            "name": "Test",
            "value": 42
        }
        
        result = self.engine.validate("extended", config)
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 2)  # Missing id and created_at


if __name__ == "__main__":
    unittest.main()