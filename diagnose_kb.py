#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to diagnose and fix the knowledge base JSON structure.
"""

import os
import json
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def diagnose_and_fix_knowledge_base():
    """
    Diagnose and fix the knowledge base structure.
    """
    kb_path = "data/knowledge_base.json"
    
    # Analyze the knowledge base structure
    try:
        with open(kb_path, "r") as f:
            kb_data = json.load(f)
        
        logger.info(f"Knowledge base data type: {type(kb_data)}")
        
        if isinstance(kb_data, dict):
            # Count different value types
            types_count = {}
            string_entries = []
            problematic_entries = []
            
            for key, value in kb_data.items():
                value_type = type(value).__name__
                types_count[value_type] = types_count.get(value_type, 0) + 1
                
                # Log a few examples of string values
                if isinstance(value, str) and len(string_entries) < 5:
                    string_entries.append((key, value))
                
                # Check if the value is a dictionary but doesn't have required fields
                if isinstance(value, dict) and ("title" not in value or not isinstance(value.get("title"), str)):
                    problematic_entries.append((key, value))
            
            logger.info(f"Knowledge base contains {len(kb_data)} entries with the following types:")
            for t, count in types_count.items():
                logger.info(f"  - {t}: {count} entries")
            
            if string_entries:
                logger.info("Examples of string entries:")
                for key, value in string_entries:
                    logger.info(f"  - {key}: {value[:50]}...")
            
            if problematic_entries:
                logger.info("Examples of problematic dictionary entries:")
                for key, value in problematic_entries[:5]:
                    logger.info(f"  - {key}: {value}")
            
            # Create a fixed knowledge base
            logger.info("Creating fixed knowledge base...")
            fixed_kb = {}
            
            for key, value in kb_data.items():
                if isinstance(value, str):
                    # Convert strings to dictionaries with title
                    fixed_kb[key] = {
                        "title": value,
                        "abstract": "",
                        "domain_id": 0,
                        "impact_score": 0.5
                    }
                elif isinstance(value, dict):
                    # Ensure dictionary has all required fields
                    fixed_entry = {"domain_id": 0, "impact_score": 0.5}
                    
                    # Copy existing fields
                    for field in value:
                        if field in ["title", "abstract", "domain_id", "impact_score"]:
                            fixed_entry[field] = value[field]
                    
                    # Ensure title exists
                    if "title" not in fixed_entry or not isinstance(fixed_entry["title"], str):
                        fixed_entry["title"] = key  # Use key as title if missing
                    
                    # Ensure abstract exists
                    if "abstract" not in fixed_entry:
                        fixed_entry["abstract"] = ""
                    
                    fixed_kb[key] = fixed_entry
                else:
                    # Skip other types
                    logger.warning(f"Skipping entry {key} with type {type(value).__name__}")
            
            # Backup original file
            backup_path = kb_path + ".bak"
            shutil.copy(kb_path, backup_path)
            logger.info(f"Original knowledge base backed up to {backup_path}")
            
            # Save fixed knowledge base
            with open(kb_path, "w") as f:
                json.dump(fixed_kb, f, indent=2)
            
            logger.info(f"Fixed knowledge base saved with {len(fixed_kb)} entries")
            
            # Verify the fixed file
            with open(kb_path, "r") as f:
                verified_kb = json.load(f)
            
            # Check for any remaining issues
            issues = 0
            for key, value in verified_kb.items():
                if not isinstance(value, dict) or "title" not in value:
                    issues += 1
            
            if issues > 0:
                logger.error(f"Found {issues} remaining issues in the fixed knowledge base")
                return False
            else:
                logger.info("Fixed knowledge base verified successfully")
                return True
            
        else:
            logger.error(f"Unexpected knowledge base format: {type(kb_data)}")
            return False
        
    except Exception as e:
        logger.error(f"Error analyzing knowledge base: {e}")
        return False

def create_test_dataset():
    """
    Create a small test dataset to verify the fix.
    """
    try:
        # Load fixed knowledge base
        with open("data/knowledge_base.json", "r") as f:
            kb_data = json.load(f)
        
        # Take a small sample for testing
        test_kb = {}
        for i, (key, value) in enumerate(kb_data.items()):
            if i >= 10:  # Just use 10 entries
                break
            test_kb[key] = value
        
        # Save test dataset
        os.makedirs("data/test", exist_ok=True)
        with open("data/test/kb_sample.json", "w") as f:
            json.dump(test_kb, f, indent=2)
        
        logger.info(f"Created test dataset with {len(test_kb)} entries")
        
        # Create a sample object with the expected structure
        papers = []
        for key, value in test_kb.items():
            paper = {
                "id": key,
                "title": value.get("title", ""),
                "abstract": value.get("abstract", ""),
                "domain_id": value.get("domain_id", 0),
                "impact_score": value.get("impact_score", 0.5)
            }
            papers.append(paper)
        
        # Save sample papers format
        with open("data/test/sample_papers.json", "w") as f:
            json.dump(papers, f, indent=2)
        
        logger.info("Created sample papers format")
        
        return True
    except Exception as e:
        logger.error(f"Error creating test dataset: {e}")
        return False

def test_knowledge_base_loading():
    """
    Test loading the knowledge base to ensure it's properly formatted.
    """
    try:
        # Function similar to the one in prepare_datasets
        kb_papers = []
        with open("data/knowledge_base.json", "r") as f:
            kb_data = json.load(f)
        
        for key, value in kb_data.items():
            try:
                # This was failing before with "'str' object has no attribute 'get'"
                # because value was a string instead of a dict
                paper = {
                    "id": key,
                    "title": value.get("title", ""),
                    "abstract": value.get("abstract", ""),
                    "domain_id": value.get("domain_id", 0),
                    "impact_score": value.get("impact_score", 0.5)
                }
                kb_papers.append(paper)
            except Exception as e:
                logger.error(f"Error processing entry {key}: {e}")
                return False
        
        logger.info(f"Successfully processed {len(kb_papers)} papers from knowledge base")
        return True
    except Exception as e:
        logger.error(f"Error testing knowledge base loading: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting knowledge base diagnosis...")
    
    # Fix the knowledge base
    if diagnose_and_fix_knowledge_base():
        logger.info("Knowledge base structure fixed successfully")
    else:
        logger.error("Failed to fix knowledge base structure")
        return
    
    # Create a test dataset
    if create_test_dataset():
        logger.info("Test dataset created successfully")
    else:
        logger.error("Failed to create test dataset")
        return
    
    # Test knowledge base loading
    if test_knowledge_base_loading():
        logger.info("Knowledge base loading test passed")
        print("\n" + "="*50)
        print("KNOWLEDGE BASE FIXED SUCCESSFULLY!")
        print("You can now run the training without errors.")
        print("="*50 + "\n")
    else:
        logger.error("Knowledge base loading test failed")

if __name__ == "__main__":
    main() 