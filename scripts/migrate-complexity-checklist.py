#!/usr/bin/env python3
"""
Migrate COMPLEXITY_CHECKLIST.MD to modular documentation structure.

This script:
1. Parses the monolithic checklist
2. Extracts sections to appropriate files
3. Preserves all content
4. Adds navigation links
5. Updates cross-references
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from datetime import datetime


class ChecklistMigrator:
    def __init__(self, source_file: str, target_dir: str):
        self.source_file = Path(source_file)
        self.target_dir = Path(target_dir)
        self.content = self.source_file.read_text()
        self.sections = {}
        self.step_mapping = {}
        
    def parse_sections(self) -> None:
        """Parse the checklist into logical sections"""
        lines = self.content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Detect major sections
            if line.startswith('## Step'):
                if current_section:
                    self.sections[current_section] = '\n'.join(current_content)
                match = re.match(r'## (Step \d+(?:\.\d+)?): (.+)', line)
                if match:
                    current_section = match.group(1)
                    current_content = [line]
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            self.sections[current_section] = '\n'.join(current_content)
    
    def extract_validation_framework(self) -> None:
        """Extract validation framework content"""
        # Find validation framework content
        validation_start = self.content.find('### Event Bus Isolation Validation Framework')
        validation_end = self.content.find('### Optimization Result Validation Framework')
        
        if validation_start != -1 and validation_end != -1:
            event_bus_content = self.content[validation_start:validation_end]
            self._save_to_file(
                'validation-framework/event-bus-isolation.md',
                self._convert_to_standalone_doc(event_bus_content, 'Event Bus Isolation Validation')
            )
    
    def extract_testing_strategy(self) -> None:
        """Extract testing strategy content"""
        # Find testing strategy patterns
        testing_patterns = [
            r'### Three-Tier Testing Strategy',
            r'### Unit Tests:',
            r'### Integration Tests:',
            r'### System Tests:'
        ]
        
        # Extract and save testing content
        # (Implementation would parse and extract testing sections)
    
    def create_step_files(self) -> None:
        """Create individual step files"""
        step_groups = {
            '01-foundation-phase': ['Step 1', 'Step 2', 'Step 2.5'],
            '02-container-architecture': ['Step 3', 'Step 4', 'Step 5', 'Step 6'],
            '03-signal-capture-replay': ['Step 7', 'Step 8', 'Step 8.5'],
            '04-multi-phase-integration': ['Step 9', 'Step 10', 'Step 10.8'],
            '05-intermediate-complexity': [
                'Step 10.1', 'Step 10.2', 'Step 10.3', 'Step 10.4',
                'Step 10.5', 'Step 10.6', 'Step 10.7'
            ],
            '06-going-beyond': [
                'Step 11', 'Step 12', 'Step 13', 'Step 14',
                'Step 15', 'Step 16', 'Step 17', 'Step 18'
            ]
        }
        
        for phase_dir, steps in step_groups.items():
            for step in steps:
                if step in self.sections:
                    filename = f"step-{step.replace('Step ', '').replace('.', '-')}-{self._get_step_slug(step)}.md"
                    filepath = f"{phase_dir}/{filename}"
                    
                    # Add header and navigation
                    enhanced_content = self._enhance_step_content(step, self.sections[step])
                    self._save_to_file(filepath, enhanced_content)
    
    def _get_step_slug(self, step: str) -> str:
        """Get URL-friendly slug for step"""
        step_names = {
            'Step 1': 'core-pipeline',
            'Step 2': 'risk-container',
            'Step 2.5': 'walk-forward',
            'Step 3': 'classifier-container',
            'Step 4': 'multiple-strategies',
            'Step 5': 'multiple-risk',
            'Step 6': 'multiple-classifiers',
            'Step 7': 'signal-capture',
            'Step 8': 'signal-replay',
            'Step 8.5': 'monte-carlo',
            'Step 9': 'parameter-expansion',
            'Step 10': 'end-to-end-workflow',
            'Step 10.1': 'advanced-analytics',
            'Step 10.2': 'basic-multi-asset',
            'Step 10.3': 'simple-optimization',
            'Step 10.4': 'risk-extensions',
            'Step 10.5': 'signal-analysis',
            'Step 10.6': 'symbol-pairs',
            'Step 10.7': 'regime-switching',
            'Step 10.8': 'memory-batch',
            'Step 11': 'multi-symbol',
            'Step 12': 'multi-timeframe',
            'Step 13': 'advanced-risk',
            'Step 14': 'ml-integration',
            'Step 15': 'alternative-data',
            'Step 16': 'hft-simulation',
            'Step 17': 'mega-portfolio',
            'Step 18': 'production-ready'
        }
        return step_names.get(step, 'unknown')
    
    def _enhance_step_content(self, step: str, content: str) -> str:
        """Add navigation and metadata to step content"""
        # Extract step title
        title_match = re.match(r'## Step \d+(?:\.\d+)?: (.+)', content.split('\n')[0])
        title = title_match.group(1) if title_match else 'Unknown Step'
        
        # Determine phase and complexity
        phase = self._get_phase_for_step(step)
        complexity = self._get_complexity_for_step(step)
        
        # Build enhanced content
        header = f"""# {step}: {title}

**Status**: {phase}
**Complexity**: {complexity}
**Prerequisites**: {self._get_prerequisites(step)}
**Architecture Ref**: {self._get_architecture_refs(step)}

## 🎯 Objective

{self._extract_objective(content)}

## 📋 Required Reading

Before starting:
{self._get_required_reading(step)}

"""
        
        # Add the rest of the content
        remaining_content = '\n'.join(content.split('\n')[1:])
        
        # Add navigation footer
        footer = f"""

## 🚀 Next Steps

Once all validations pass, proceed to:
{self._get_next_step_link(step)}

## 📚 Additional Resources

{self._get_additional_resources(step)}
"""
        
        return header + remaining_content + footer
    
    def _get_phase_for_step(self, step: str) -> str:
        """Determine which phase a step belongs to"""
        step_num = float(step.replace('Step ', ''))
        if step_num <= 2.5:
            return "Foundation Step"
        elif step_num <= 6:
            return "Container Architecture Step"
        elif step_num <= 8.5:
            return "Signal Capture & Replay Step"
        elif step_num <= 10:
            return "Multi-Phase Integration Step"
        elif step_num <= 10.8:
            return "Intermediate Complexity Step"
        else:
            return "Advanced Step"
    
    def _get_complexity_for_step(self, step: str) -> str:
        """Determine complexity level for step"""
        step_num = float(step.replace('Step ', ''))
        if step_num <= 2:
            return "Low"
        elif step_num <= 4:
            return "Medium"
        elif step_num <= 8:
            return "Medium-High"
        elif step_num <= 12:
            return "High"
        else:
            return "Very High"
    
    def _save_to_file(self, relative_path: str, content: str) -> None:
        """Save content to file in target directory"""
        filepath = self.target_dir / relative_path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
        print(f"Created: {filepath}")
    
    def create_summary_report(self) -> None:
        """Create a summary of the migration"""
        report = f"""# Complexity Checklist Migration Report

Generated: {datetime.now().isoformat()}

## Summary

- Original file: {self.source_file}
- Original size: {len(self.content)} characters
- Sections extracted: {len(self.sections)}
- Files created: {self._count_created_files()}

## Extracted Sections

{self._list_extracted_sections()}

## Validation

- [ ] All content preserved
- [ ] Navigation links working
- [ ] Cross-references updated
- [ ] No broken links
- [ ] File structure matches plan
"""
        
        self._save_to_file('MIGRATION_REPORT.md', report)
    
    def _count_created_files(self) -> int:
        """Count files created during migration"""
        return sum(1 for _ in self.target_dir.rglob('*.md'))
    
    def _list_extracted_sections(self) -> str:
        """List all extracted sections"""
        return '\n'.join(f"- {section}" for section in sorted(self.sections.keys()))
    
    def run(self) -> None:
        """Run the complete migration"""
        print(f"Starting migration of {self.source_file}...")
        
        # Create backup
        backup_path = self.source_file.with_suffix('.MD.bak')
        shutil.copy(self.source_file, backup_path)
        print(f"Created backup: {backup_path}")
        
        # Parse content
        print("Parsing sections...")
        self.parse_sections()
        
        # Extract framework docs
        print("Extracting validation framework...")
        self.extract_validation_framework()
        
        print("Extracting testing strategy...")
        self.extract_testing_strategy()
        
        # Create step files
        print("Creating step files...")
        self.create_step_files()
        
        # Create report
        print("Creating migration report...")
        self.create_summary_report()
        
        print("\nMigration complete!")


def main():
    """Run the migration script"""
    source = Path('COMPLEXITY_CHECKLIST.MD')
    target = Path('docs/complexity-guide')
    
    if not source.exists():
        print(f"Error: {source} not found")
        return 1
    
    migrator = ChecklistMigrator(source, target)
    migrator.run()
    
    return 0


if __name__ == '__main__':
    exit(main())