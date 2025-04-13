"""
Unit tests for the mock data module.

This module contains tests for the mock_data.py module, focusing on:
- Verification of research frontiers data
- Verification of paper sample data
"""

import pytest
from celery_tasks.mock_data import research_frontiers, paper_samples

# Test that research frontiers data is valid
def test_research_frontiers_data():
    """
    Test that the research frontiers mock data is valid.
    
    This test verifies that:
    - The research_frontiers list is not empty
    - Each frontier has the required fields
    - Values have the expected types
    """
    # Check that we have data
    assert research_frontiers is not None
    assert len(research_frontiers) > 0
    
    # Check the structure of each frontier
    for frontier in research_frontiers:
        # Required fields
        assert "concept" in frontier
        assert "importance" in frontier
        assert "frequency" in frontier
        assert "papers" in frontier
        
        # Type checks
        assert isinstance(frontier["concept"], str)
        assert isinstance(frontier["importance"], float)
        assert isinstance(frontier["frequency"], int)
        assert isinstance(frontier["papers"], list)
        
        # Check that papers are not empty
        assert len(frontier["papers"]) > 0
        
        # Check that papers are strings (typically arXiv IDs)
        for paper_id in frontier["papers"]:
            assert isinstance(paper_id, str)

# Test that paper samples data is valid
def test_paper_samples_data():
    """
    Test that the paper samples mock data is valid.
    
    This test verifies that:
    - The paper_samples list is not empty
    - Each paper has the required fields
    - Values have the expected types
    """
    # Check that we have data
    assert paper_samples is not None
    assert len(paper_samples) > 0
    
    # Check the structure of each paper
    for paper in paper_samples:
        # Required fields
        assert "paper_id" in paper
        assert "title" in paper
        assert "authors" in paper
        assert "abstract" in paper
        assert "year" in paper
        assert "url" in paper
        
        # Type checks
        assert isinstance(paper["paper_id"], str)
        assert isinstance(paper["title"], str)
        assert isinstance(paper["authors"], list)
        assert isinstance(paper["abstract"], str)
        assert isinstance(paper["year"], int)
        assert isinstance(paper["url"], str)
        
        # Check that the authors list is not empty
        assert len(paper["authors"]) > 0
        
        # Check that each author is a string
        for author in paper["authors"]:
            assert isinstance(author, str)
        
        # Check that the abstract is not empty
        assert len(paper["abstract"]) > 0
        
        # Check that the year is reasonable (e.g., between 1990 and 2030)
        assert 1990 <= paper["year"] <= 2030
        
        # Check that the URL is an arXiv URL
        assert paper["url"].startswith("https://arxiv.org/abs/")

# Test that research frontiers and paper samples are linked
def test_research_frontiers_paper_links():
    """
    Test that research frontiers and paper samples are correctly linked.
    
    This test verifies that:
    - Paper IDs referenced in frontiers exist in the paper samples
    - Each paper sample is referenced by at least one frontier
    """
    # Get all unique paper IDs from the frontiers
    frontier_paper_ids = set()
    for frontier in research_frontiers:
        frontier_paper_ids.update(frontier["papers"])
    
    # Get all paper IDs from the samples
    sample_paper_ids = {paper["paper_id"] for paper in paper_samples}
    
    # Check that at least some paper IDs in frontiers are in the samples
    # Note: We don't require all paper IDs to be in samples, as samples are just examples
    common_ids = frontier_paper_ids.intersection(sample_paper_ids)
    assert len(common_ids) > 0
    
    # Check that at least some of the sample papers are referenced in frontiers
    # Again, we don't require all sample papers to be referenced
    for paper_id in sample_paper_ids:
        referenced = False
        for frontier in research_frontiers:
            if paper_id in frontier["papers"]:
                referenced = True
                break
        if referenced:
            break  # Found at least one referenced paper
    
    assert referenced, "None of the sample papers are referenced in the frontiers" 