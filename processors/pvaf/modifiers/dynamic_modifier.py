"""
Modifier System: Dynamic Modifiers

Applies contextual risk modifiers (mutations, vendor responses, citations, etc.)
to the base PVAF score to compute the final score.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)


def apply_dynamic_modifiers(base_score: float, vulnerability_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Apply dynamic modifiers to the base PVAF score.
    
    Parameters:
    - base_score: The calculated base PVAF score (0-100)
    - vulnerability_data: Dict containing modifier-related fields:
        - mutations: Data about variations and mutations
        - corporate_response: Information about vendor responses
        - academic_citations: Academic research mentioning this vulnerability
        - tool_integration: Tool/framework integration status
        - regulatory_attention: Regulatory/media attention data
        
    Returns:
    - modified_score: Adjusted PVAF score
    - applied_modifiers: Dict of applied modifiers and their values
    """
    applied_modifiers = {}
    modified_score = float(base_score)
    
    logger.debug(f"Applying dynamic modifiers to base score: {base_score}")
    
    # Mutation Factor (+5 to +15)
    if 'mutations' in vulnerability_data and vulnerability_data['mutations']:
        mutation_modifier = _calculate_mutation_modifier(vulnerability_data['mutations'])
        if mutation_modifier > 0:
            applied_modifiers['mutation_factor'] = mutation_modifier
            modified_score += mutation_modifier
            logger.debug(f"Applied mutation modifier: +{mutation_modifier}")
    
    # Corporate Response (-5 to -20)
    if 'corporate_response' in vulnerability_data and vulnerability_data['corporate_response']:
        response_modifier = _calculate_response_modifier(vulnerability_data['corporate_response'])
        if response_modifier != 0:  # Can be negative or positive in edge cases
            applied_modifiers['corporate_response'] = response_modifier
            modified_score += response_modifier
            logger.debug(f"Applied corporate response modifier: {response_modifier}")
    
    # Academic Citation (+10)
    if 'academic_citations' in vulnerability_data and vulnerability_data['academic_citations']:
        if _has_valid_academic_citations(vulnerability_data['academic_citations']):
            applied_modifiers['academic_citation'] = 10
            modified_score += 10
            logger.debug("Applied academic citation modifier: +10")
    
    # Tool Integration (+15)
    if 'tool_integration' in vulnerability_data and vulnerability_data['tool_integration']:
        tool_modifier = _calculate_tool_integration_modifier(vulnerability_data['tool_integration'])
        if tool_modifier > 0:
            applied_modifiers['tool_integration'] = tool_modifier
            modified_score += tool_modifier
            logger.debug(f"Applied tool integration modifier: +{tool_modifier}")
    
    # Regulatory Attention (+10)
    if 'regulatory_attention' in vulnerability_data and vulnerability_data['regulatory_attention']:
        regulatory_modifier = _calculate_regulatory_attention_modifier(vulnerability_data['regulatory_attention'])
        if regulatory_modifier > 0:
            applied_modifiers['regulatory_attention'] = regulatory_modifier
            modified_score += regulatory_modifier
            logger.debug(f"Applied regulatory attention modifier: +{regulatory_modifier}")
    
    # Media Coverage Modifier (+5 to +12)
    if 'media_coverage' in vulnerability_data and vulnerability_data['media_coverage']:
        media_modifier = _calculate_media_coverage_modifier(vulnerability_data['media_coverage'])
        if media_modifier > 0:
            applied_modifiers['media_coverage'] = media_modifier
            modified_score += media_modifier
            logger.debug(f"Applied media coverage modifier: +{media_modifier}")
    
    # Exploit Maturity Modifier (+5 to +20)
    if 'exploit_maturity' in vulnerability_data and vulnerability_data['exploit_maturity']:
        maturity_modifier = _calculate_exploit_maturity_modifier(vulnerability_data['exploit_maturity'])
        if maturity_modifier > 0:
            applied_modifiers['exploit_maturity'] = maturity_modifier
            modified_score += maturity_modifier
            logger.debug(f"Applied exploit maturity modifier: +{maturity_modifier}")
    
    # Underground Activity Modifier (+8 to +25)
    if 'underground_activity' in vulnerability_data and vulnerability_data['underground_activity']:
        underground_modifier = _calculate_underground_activity_modifier(vulnerability_data['underground_activity'])
        if underground_modifier > 0:
            applied_modifiers['underground_activity'] = underground_modifier
            modified_score += underground_modifier
            logger.debug(f"Applied underground activity modifier: +{underground_modifier}")
    
    # Age/Staleness Modifier (-5 to -15)
    if 'temporal_data' in vulnerability_data and vulnerability_data['temporal_data']:
        age_modifier = _calculate_age_modifier(vulnerability_data['temporal_data'])
        if age_modifier < 0:
            applied_modifiers['age_staleness'] = age_modifier
            modified_score += age_modifier
            logger.debug(f"Applied age/staleness modifier: {age_modifier}")
    
    # Cross-Reference Validation Modifier (+3 to +8)
    cross_ref_modifier = _calculate_cross_reference_modifier(vulnerability_data)
    if cross_ref_modifier > 0:
        applied_modifiers['cross_reference_validation'] = cross_ref_modifier
        modified_score += cross_ref_modifier
        logger.debug(f"Applied cross-reference validation modifier: +{cross_ref_modifier}")
    
    # Ensure final score stays within bounds
    modified_score = max(0, min(modified_score, 100))
    
    logger.info(f"Dynamic modifiers applied: {base_score} -> {modified_score} (modifiers: {applied_modifiers})")
    
    return modified_score, applied_modifiers


def _calculate_mutation_modifier(mutation_data: Dict[str, Any]) -> float:
    """
    Calculate modifier based on vulnerability mutations.
    
    Parameters:
    - mutation_data: Dict containing:
        - variants: List of mutation variants
        - diversity_score: Float indicating mutation diversity (0-1)
        - innovation_level: String indicating level of innovation
        - functional_changes: List of functional changes across variants
        
    Returns:
    - Float modifier from 0 to +15 based on number and diversity of mutations
    """
    if not isinstance(mutation_data, dict):
        return 0
    
    variants = mutation_data.get('variants', [])
    diversity_score = mutation_data.get('diversity_score', 0)
    innovation_level = mutation_data.get('innovation_level', 'low')
    functional_changes = mutation_data.get('functional_changes', [])
    
    variant_count = len(variants) if isinstance(variants, list) else 0
    
    # Base modifier from variant count
    if variant_count == 0:
        base_modifier = 0
    elif variant_count <= 2:
        base_modifier = 3  # Minimal mutations
    elif variant_count <= 5:
        base_modifier = 7  # Moderate mutations
    elif variant_count <= 10:
        base_modifier = 12  # Significant mutations
    else:
        base_modifier = 15  # Extensive mutations
    
    # Diversity bonus (0-3 points)
    diversity_bonus = 0
    if isinstance(diversity_score, (int, float)) and diversity_score > 0.5:
        diversity_bonus = min(3, int(diversity_score * 3))
    
    # Innovation bonus (0-5 points)
    innovation_bonus = 0
    if innovation_level == 'high':
        innovation_bonus = 5
    elif innovation_level == 'medium':
        innovation_bonus = 3
    elif innovation_level == 'low':
        innovation_bonus = 1
    
    # Functional change bonus (0-2 points)
    functional_bonus = 0
    if isinstance(functional_changes, list) and len(functional_changes) > 0:
        functional_bonus = min(2, len(functional_changes))
    
    total_modifier = min(15, base_modifier + diversity_bonus + innovation_bonus + functional_bonus)
    
    logger.debug(f"Mutation modifier calculation: variants={variant_count}, "
                f"diversity={diversity_score}, innovation={innovation_level} -> {total_modifier}")
    
    return total_modifier


def _calculate_response_modifier(response_data: Dict[str, Any]) -> float:
    """
    Calculate modifier based on corporate response.
    
    Parameters:
    - response_data: Dict containing:
        - response_type: String indicating type of response
        - effectiveness: Float (0-1) indicating response effectiveness
        - response_speed: Days from disclosure to response
        - affected_models: List of models that were patched
        - public_acknowledgment: Boolean indicating public acknowledgment
        
    Returns:
    - Float modifier from -20 to +5 based on response effectiveness
    """
    if not isinstance(response_data, dict):
        return 0
    
    response_type = response_data.get('response_type', 'none')
    effectiveness = response_data.get('effectiveness', 0)
    response_speed = response_data.get('response_speed', float('inf'))
    affected_models = response_data.get('affected_models', [])
    public_acknowledgment = response_data.get('public_acknowledgment', False)
    
    # Base modifier based on response type and effectiveness
    base_modifier = 0
    
    if response_type == 'none' or effectiveness == 0:
        base_modifier = 0
    elif response_type == 'acknowledgment' and effectiveness < 0.2:
        base_modifier = -3  # Minimal acknowledgment
    elif response_type == 'investigation' and effectiveness < 0.4:
        base_modifier = -5  # Under investigation
    elif response_type == 'partial_mitigation' and effectiveness < 0.6:
        base_modifier = -8  # Partial fix implemented
    elif response_type == 'specific_mitigation' and effectiveness < 0.8:
        base_modifier = -12  # Targeted mitigation
    elif response_type == 'comprehensive_fix' or effectiveness >= 0.8:
        base_modifier = -18  # Comprehensive solution
    
    # Speed bonus/penalty (-2 to +2)
    speed_modifier = 0
    if isinstance(response_speed, (int, float)) and response_speed != float('inf'):
        if response_speed <= 7:  # Very fast response (within a week)
            speed_modifier = -2
        elif response_speed <= 30:  # Fast response (within a month)
            speed_modifier = -1
        elif response_speed <= 90:  # Normal response
            speed_modifier = 0
        elif response_speed <= 180:  # Slow response
            speed_modifier = 1
        else:  # Very slow response
            speed_modifier = 2
    
    # Model coverage modifier (-3 to +1)
    coverage_modifier = 0
    if isinstance(affected_models, list):
        if len(affected_models) >= 5:  # Comprehensive coverage
            coverage_modifier = -3
        elif len(affected_models) >= 3:  # Good coverage
            coverage_modifier = -2
        elif len(affected_models) >= 1:  # Limited coverage
            coverage_modifier = -1
        else:  # No coverage
            coverage_modifier = 1
    
    # Public acknowledgment modifier (-1 to +1)
    acknowledgment_modifier = -1 if public_acknowledgment else 1
    
    # Special cases
    if response_type == 'denial' or response_type == 'dismissal':
        base_modifier = 3  # Denial actually increases risk
    elif response_type == 'ineffective_response':
        base_modifier = 2  # Ineffective response can increase attention
    
    total_modifier = base_modifier + speed_modifier + coverage_modifier + acknowledgment_modifier
    
    # Cap the modifier
    total_modifier = max(-20, min(5, total_modifier))
    
    logger.debug(f"Corporate response modifier: type={response_type}, "
                f"effectiveness={effectiveness} -> {total_modifier}")
    
    return total_modifier


def _has_valid_academic_citations(citation_data: Dict[str, Any]) -> bool:
    """
    Check if vulnerability has valid academic citations.
    
    Parameters:
    - citation_data: Dict containing:
        - papers: List of academic papers mentioning this vulnerability
        - conferences: List of conferences where it was discussed
        - peer_reviewed: Boolean indicating peer-reviewed status
        - citation_count: Number of citations
        - venue_quality: Quality rating of publication venues
        
    Returns:
    - Boolean indicating whether there are valid academic citations
    """
    if not isinstance(citation_data, dict):
        return False
    
    papers = citation_data.get('papers', [])
    conferences = citation_data.get('conferences', [])
    peer_reviewed = citation_data.get('peer_reviewed', False)
    citation_count = citation_data.get('citation_count', 0)
    venue_quality = citation_data.get('venue_quality', 'unknown')
    
    # Validate papers
    valid_papers = 0
    if isinstance(papers, list):
        for paper in papers:
            if isinstance(paper, dict):
                # Check for required fields
                if (paper.get('title') and paper.get('authors') and 
                    paper.get('venue') and paper.get('year')):
                    # Check venue quality
                    venue = paper.get('venue', '').lower()
                    if any(quality_venue in venue for quality_venue in [
                        'ieee', 'acm', 'usenix', 'neurips', 'icml', 'iclr', 
                        'aaai', 'ijcai', 'security', 'crypto', 'privacy'
                    ]):
                        valid_papers += 1
            elif isinstance(paper, str) and len(paper) > 20:  # Simple string citation
                valid_papers += 1
    
    # Validate conferences
    valid_conferences = 0
    if isinstance(conferences, list):
        quality_conferences = [
            'defcon', 'blackhat', 'rsa', 'bsides', 'owasp', 'ieee security',
            'acm ccs', 'usenix security', 'neurips', 'icml', 'iclr'
        ]
        for conf in conferences:
            if isinstance(conf, str):
                conf_lower = conf.lower()
                if any(quality_conf in conf_lower for quality_conf in quality_conferences):
                    valid_conferences += 1
    
    # Determine validity
    has_quality_papers = valid_papers >= 1 and peer_reviewed
    has_quality_conferences = valid_conferences >= 1
    has_sufficient_citations = citation_count >= 3
    has_quality_venue = venue_quality in ['high', 'top-tier']
    
    is_valid = (has_quality_papers or has_quality_conferences or 
                has_sufficient_citations or has_quality_venue)
    
    logger.debug(f"Academic citation validation: papers={valid_papers}, "
                f"conferences={valid_conferences}, peer_reviewed={peer_reviewed} -> {is_valid}")
    
    return is_valid


def _calculate_tool_integration_modifier(tool_data: Dict[str, Any]) -> float:
    """
    Check if vulnerability has been integrated into tools/frameworks.
    
    Parameters:
    - tool_data: Dict containing:
        - integrated_tools: List of tools that include this vulnerability
        - framework_support: List of frameworks supporting this
        - automation_level: Level of automation (manual, semi-auto, full-auto)
        - public_availability: Whether tools are publicly available
        
    Returns:
    - Float modifier from 0 to +20 based on integration level
    """
    if not isinstance(tool_data, dict):
        return 0
    
    integrated_tools = tool_data.get('integrated_tools', [])
    framework_support = tool_data.get('framework_support', [])
    automation_level = tool_data.get('automation_level', 'manual')
    public_availability = tool_data.get('public_availability', False)
    
    modifier = 0
    
    # Base modifier for tool integration
    tool_count = len(integrated_tools) if isinstance(integrated_tools, list) else 0
    if tool_count >= 3:
        modifier += 12  # Multiple tool integration
    elif tool_count >= 1:
        modifier += 8   # Some tool integration
    
    # Framework support bonus
    framework_count = len(framework_support) if isinstance(framework_support, list) else 0
    if framework_count >= 2:
        modifier += 5   # Multiple framework support
    elif framework_count >= 1:
        modifier += 3   # Single framework support
    
    # Automation level bonus
    if automation_level == 'full-auto':
        modifier += 5   # Fully automated
    elif automation_level == 'semi-auto':
        modifier += 3   # Semi-automated
    elif automation_level == 'manual':
        modifier += 1   # Manual implementation
    
    # Public availability bonus
    if public_availability:
        modifier += 3   # Publicly available tools
    
    # Check for specific high-impact tools
    if isinstance(integrated_tools, list):
        high_impact_tools = [
            'metasploit', 'burp suite', 'owasp zap', 'nuclei', 
            'sqlmap', 'gobuster', 'ffuf', 'custom gpt'
        ]
        for tool in integrated_tools:
            if isinstance(tool, str) and any(hit in tool.lower() for hit in high_impact_tools):
                modifier += 2  # Bonus for high-impact tools
                break
    
    # Cap the modifier
    modifier = min(20, modifier)
    
    logger.debug(f"Tool integration modifier: tools={tool_count}, "
                f"frameworks={framework_count}, automation={automation_level} -> {modifier}")
    
    return modifier


def _calculate_regulatory_attention_modifier(attention_data: Dict[str, Any]) -> float:
    """
    Check if vulnerability has received regulatory/media attention.
    
    Parameters:
    - attention_data: Dict containing:
        - regulatory_mentions: List of regulatory body mentions
        - media_coverage: List of media coverage instances
        - government_alerts: List of government security alerts
        - industry_advisories: List of industry security advisories
        
    Returns:
    - Float modifier from 0 to +15 based on attention level
    """
    if not isinstance(attention_data, dict):
        return 0
    
    regulatory_mentions = attention_data.get('regulatory_mentions', [])
    media_coverage = attention_data.get('media_coverage', [])
    government_alerts = attention_data.get('government_alerts', [])
    industry_advisories = attention_data.get('industry_advisories', [])
    
    modifier = 0
    
    # Regulatory mentions
    if isinstance(regulatory_mentions, list) and len(regulatory_mentions) > 0:
        high_profile_regulators = ['nist', 'cisa', 'fcc', 'ftc', 'sec', 'eu', 'gdpr']
        for mention in regulatory_mentions:
            if isinstance(mention, str):
                mention_lower = mention.lower()
                if any(reg in mention_lower for reg in high_profile_regulators):
                    modifier += 5  # High-profile regulatory attention
                    break
        else:
            modifier += 3  # General regulatory attention
    
    # Government alerts
    if isinstance(government_alerts, list) and len(government_alerts) > 0:
        modifier += 6  # Government security alerts are significant
    
    # Industry advisories
    if isinstance(industry_advisories, list) and len(industry_advisories) > 0:
        modifier += 4  # Industry recognition
    
    # Media coverage (handled separately in media coverage modifier)
    # But we can add a small bonus here for mainstream media
    if isinstance(media_coverage, list):
        mainstream_media = ['cnn', 'bbc', 'reuters', 'bloomberg', 'wsj', 'nytimes']
        for coverage in media_coverage:
            if isinstance(coverage, str):
                coverage_lower = coverage.lower()
                if any(media in coverage_lower for media in mainstream_media):
                    modifier += 2  # Mainstream media attention
                    break
    
    # Cap the modifier
    modifier = min(15, modifier)
    
    logger.debug(f"Regulatory attention modifier: regulatory={len(regulatory_mentions)}, "
                f"gov_alerts={len(government_alerts)} -> {modifier}")
    
    return modifier


def _calculate_media_coverage_modifier(media_data: Dict[str, Any]) -> float:
    """
    Calculate modifier based on media coverage.
    
    Parameters:
    - media_data: Dict containing media coverage information
    
    Returns:
    - Float modifier from 0 to +12 based on coverage scope and quality
    """
    if not isinstance(media_data, dict):
        return 0
    
    coverage_count = media_data.get('coverage_count', 0)
    media_tier = media_data.get('media_tier', 'low')  # low, medium, high
    international_coverage = media_data.get('international_coverage', False)
    
    modifier = 0
    
    # Base modifier from coverage count
    if coverage_count >= 10:
        modifier += 6
    elif coverage_count >= 5:
        modifier += 4
    elif coverage_count >= 2:
        modifier += 2
    elif coverage_count >= 1:
        modifier += 1
    
    # Media tier bonus
    if media_tier == 'high':
        modifier += 4
    elif media_tier == 'medium':
        modifier += 2
    
    # International coverage bonus
    if international_coverage:
        modifier += 2
    
    return min(12, modifier)


def _calculate_exploit_maturity_modifier(maturity_data: Dict[str, Any]) -> float:
    """
    Calculate modifier based on exploit maturity.
    
    Parameters:
    - maturity_data: Dict containing exploit maturity information
    
    Returns:
    - Float modifier from 0 to +20 based on maturity level
    """
    if not isinstance(maturity_data, dict):
        return 0
    
    maturity_level = maturity_data.get('maturity_level', 'proof_of_concept')
    weaponization = maturity_data.get('weaponization', False)
    automation_available = maturity_data.get('automation_available', False)
    
    modifier = 0
    
    # Base modifier from maturity level
    maturity_scores = {
        'proof_of_concept': 3,
        'functional_exploit': 8,
        'reliable_exploit': 12,
        'weaponized': 18,
        'mass_deployment': 20
    }
    
    modifier += maturity_scores.get(maturity_level, 0)
    
    # Weaponization bonus
    if weaponization:
        modifier += 5
    
    # Automation bonus
    if automation_available:
        modifier += 3
    
    return min(20, modifier)


def _calculate_underground_activity_modifier(underground_data: Dict[str, Any]) -> float:
    """
    Calculate modifier based on underground/criminal activity.
    
    Parameters:
    - underground_data: Dict containing underground activity information
    
    Returns:
    - Float modifier from 0 to +25 based on underground adoption
    """
    if not isinstance(underground_data, dict):
        return 0
    
    dark_web_mentions = underground_data.get('dark_web_mentions', 0)
    criminal_use = underground_data.get('criminal_use', False)
    monetization = underground_data.get('monetization', False)
    
    modifier = 0
    
    # Dark web mentions
    if dark_web_mentions >= 10:
        modifier += 15
    elif dark_web_mentions >= 5:
        modifier += 10
    elif dark_web_mentions >= 1:
        modifier += 5
    
    # Criminal use
    if criminal_use:
        modifier += 12
    
    # Monetization
    if monetization:
        modifier += 8
    
    return min(25, modifier)


def _calculate_age_modifier(temporal_data: Dict[str, Any]) -> float:
    """
    Calculate modifier based on vulnerability age.
    
    Parameters:
    - temporal_data: Dict containing temporal information
    
    Returns:
    - Float modifier from -15 to 0 based on age (negative for old vulnerabilities)
    """
    if not isinstance(temporal_data, dict):
        return 0
    
    first_seen = temporal_data.get('first_seen')
    last_activity = temporal_data.get('last_activity')
    
    if not first_seen:
        return 0
    
    try:
        if isinstance(first_seen, str):
            first_seen_dt = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
        else:
            first_seen_dt = first_seen
        
        days_old = (datetime.now(timezone.utc) - first_seen_dt).days
        
        # Age-based modifier (newer is better for relevance)
        if days_old <= 30:
            return 0  # Very recent
        elif days_old <= 90:
            return -2  # Recent
        elif days_old <= 180:
            return -5  # Somewhat old
        elif days_old <= 365:
            return -8  # Old
        else:
            return -12  # Very old
    
    except (ValueError, TypeError):
        return 0


def _calculate_cross_reference_modifier(vulnerability_data: Dict[str, Any]) -> float:
    """
    Calculate modifier based on cross-reference validation.
    
    Parameters:
    - vulnerability_data: Full vulnerability data for cross-validation
    
    Returns:
    - Float modifier from 0 to +8 based on validation confidence
    """
    modifier = 0
    
    # Check for multiple independent sources
    sources = vulnerability_data.get('sources', [])
    if isinstance(sources, list) and len(sources) >= 3:
        modifier += 3
    elif isinstance(sources, list) and len(sources) >= 2:
        modifier += 2
    
    # Check for verification across platforms
    platforms = vulnerability_data.get('platforms', [])
    if isinstance(platforms, list) and len(platforms) >= 3:
        modifier += 2
    
    # Check for technical validation
    if vulnerability_data.get('technical_validation', False):
        modifier += 2
    
    # Check for community consensus
    community_score = vulnerability_data.get('community_validation_score', 0)
    if community_score >= 0.8:
        modifier += 1
    
    return min(8, modifier)


# Utility functions for testing and validation
def validate_modifier_data(vulnerability_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate that vulnerability data contains proper structure for modifiers.
    
    Returns:
    - Dict mapping modifier types to list of validation issues
    """
    issues = {}
    
    # Check mutations data
    if 'mutations' in vulnerability_data:
        mutation_issues = []
        mutations = vulnerability_data['mutations']
        if not isinstance(mutations, dict):
            mutation_issues.append("mutations should be a dict")
        else:
            if 'variants' in mutations and not isinstance(mutations['variants'], list):
                mutation_issues.append("mutations.variants should be a list")
        if mutation_issues:
            issues['mutations'] = mutation_issues
    
    # Check corporate response data
    if 'corporate_response' in vulnerability_data:
        response_issues = []
        response = vulnerability_data['corporate_response']
        if not isinstance(response, dict):
            response_issues.append("corporate_response should be a dict")
        else:
            if 'effectiveness' in response:
                eff = response['effectiveness']
                if not isinstance(eff, (int, float)) or not (0 <= eff <= 1):
                    response_issues.append("effectiveness should be float between 0-1")
        if response_issues:
            issues['corporate_response'] = response_issues
    
    # Additional validation can be added here
    
    return issues


def get_modifier_summary(applied_modifiers: Dict[str, float]) -> str:
    """
    Generate a human-readable summary of applied modifiers.
    
    Parameters:
    - applied_modifiers: Dict of modifier names and values
    
    Returns:
    - String summary of modifiers
    """
    if not applied_modifiers:
        return "No modifiers applied"
    
    positive_mods = [(k, v) for k, v in applied_modifiers.items() if v > 0]
    negative_mods = [(k, v) for k, v in applied_modifiers.items() if v < 0]
    
    summary_parts = []
    
    if positive_mods:
        pos_total = sum(v for _, v in positive_mods)
        pos_names = [k.replace('_', ' ') for k, _ in positive_mods]
        summary_parts.append(f"Risk increased by {pos_total:.1f} points due to: {', '.join(pos_names)}")
    
    if negative_mods:
        neg_total = sum(v for _, v in negative_mods)
        neg_names = [k.replace('_', ' ') for k, _ in negative_mods]
        summary_parts.append(f"Risk decreased by {abs(neg_total):.1f} points due to: {', '.join(neg_names)}")
    
    return "; ".join(summary_parts)


# Example usage and testing
if __name__ == "__main__":
    # Example vulnerability data for testing
    test_vulnerability = {
        'mutations': {
            'variants': ['variant1', 'variant2', 'variant3', 'variant4'],
            'diversity_score': 0.7,
            'innovation_level': 'medium',
            'functional_changes': ['obfuscation', 'role_play']
        },
        'corporate_response': {
            'response_type': 'partial_mitigation',
            'effectiveness': 0.5,
            'response_speed': 21,  # days
            'affected_models': ['gpt-4', 'claude-3'],
            'public_acknowledgment': True
        },
        'academic_citations': {
            'papers': [
                {
                    'title': 'Advanced Prompt Injection Techniques',
                    'authors': 'Smith et al.',
                    'venue': 'IEEE Security and Privacy',
                    'year': 2024
                }
            ],
            'peer_reviewed': True,
            'citation_count': 5
        },
        'tool_integration': {
            'integrated_tools': ['custom_framework', 'red_team_toolkit'],
            'automation_level': 'semi-auto',
            'public_availability': True
        },
        'regulatory_attention': {
            'regulatory_mentions': ['NIST advisory'],
            'government_alerts': ['CISA alert'],
            'media_coverage': ['TechCrunch article']
        }
    }
    
    # Test the modifier system
    base_score = 65.0
    final_score, modifiers = apply_dynamic_modifiers(base_score, test_vulnerability)
    
    print(f"Base PVAF Score: {base_score}")
    print(f"Final PVAF Score: {final_score}")
    print(f"Applied Modifiers: {modifiers}")
    print(f"Total Modifier Impact: {final_score - base_score:+.1f}")
    print(f"\nModifier Summary:")
    print(get_modifier_summary(modifiers))
    
    # Validation test
    print(f"\nValidation Issues:")
    validation_issues = validate_modifier_data(test_vulnerability)
    if validation_issues:
        for modifier_type, issues in validation_issues.items():
            print(f"  {modifier_type}: {issues}")
    else:
        print("  No validation issues found")
    
    # Test with minimal data
    print(f"\n{'='*50}")
    print("Testing with minimal vulnerability data:")
    
    minimal_vulnerability = {
        'temporal_data': {
            'first_seen': '2024-01-01T00:00:00Z'
        }
    }
    
    minimal_final_score, minimal_modifiers = apply_dynamic_modifiers(50.0, minimal_vulnerability)
    print(f"Minimal Data - Base: 50.0, Final: {minimal_final_score}, Modifiers: {minimal_modifiers}")
    
    # Test edge cases
    print(f"\nTesting edge cases:")
    
    # Empty data
    empty_score, empty_mods = apply_dynamic_modifiers(75.0, {})
    print(f"Empty data: {empty_score} (should equal base score)")
    
    # Invalid data types
    invalid_data = {
        'mutations': 'invalid_string',
        'corporate_response': ['invalid_list'],
        'academic_citations': 123
    }
    invalid_score, invalid_mods = apply_dynamic_modifiers(60.0, invalid_data)
    print(f"Invalid data types: {invalid_score} (modifiers: {invalid_mods})")
    
    # Test boundary conditions
    print(f"\nTesting boundary conditions:")
    
    # Very high base score
    high_base = 95.0
    high_vulnerability = {
        'mutations': {
            'variants': ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
            'diversity_score': 0.9,
            'innovation_level': 'high'
        },
        'tool_integration': {
            'integrated_tools': ['tool1', 'tool2', 'tool3', 'metasploit'],
            'automation_level': 'full-auto',
            'public_availability': True
        }
    }
    high_final, high_mods = apply_dynamic_modifiers(high_base, high_vulnerability)
    print(f"High base score: {high_base} -> {high_final} (should cap at 100)")
    
    # Very low base score with negative modifiers
    low_base = 5.0
    low_vulnerability = {
        'corporate_response': {
            'response_type': 'comprehensive_fix',
            'effectiveness': 0.95,
            'response_speed': 3,
            'affected_models': ['gpt-4', 'claude-3', 'bard', 'llama-2', 'palm'],
            'public_acknowledgment': True
        },
        'temporal_data': {
            'first_seen': '2020-01-01T00:00:00Z'  # Very old
        }
    }
    low_final, low_mods = apply_dynamic_modifiers(low_base, low_vulnerability)
    print(f"Low base score with negative modifiers: {low_base} -> {low_final} (should not go below 0)")
    
    print(f"\n{'='*50}")
    print("Dynamic Modifier System Test Complete")