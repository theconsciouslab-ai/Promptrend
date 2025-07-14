import math

"""
Dimension: Community Adoption

Measures how widely a technique is being shared and verified within public discourse.
"""

def calculate(vulnerability_data):
    # monitoring_data must be embedded in vulnerability_data (normalize access)
    monitoring_data = vulnerability_data.get("monitoring_data", {})
    
    # If no monitoring_data, try to extract from metadata.social_signals
    if not monitoring_data:
        social_signals = vulnerability_data.get("metadata", {}).get("social_signals", {})
        if social_signals:
            monitoring_data = {
                'mention_count': social_signals.get('cross_references', {}).get('platform_mentions', {}),
                'engagement_metrics': social_signals.get('engagement_metrics', {}),
                'success_reports': social_signals.get('community_validation', {}).get('success_confirmations', 0)
            }
    
    return calculate_community_adoption(vulnerability_data, monitoring_data)


def calculate_community_adoption(vulnerability_data, monitoring_data):
    """
    Calculate Community Adoption score (0-100) based on monitoring data.
    
    Parameters:
    - vulnerability_data: Dict containing basic vulnerability info
    - monitoring_data: Dict containing social metrics:
        - mention_count: Dict of mentions by platform
        - engagement_metrics: Dict of engagement data
        - success_reports: Number of confirmed successes
        - timeframe: Days since first observation
        
    Returns:
    - adoption_score: Integer score from 0-100
    - evidence: Dict with justification for the score
    """
    mention_data = monitoring_data.get('mention_count', {})
    engagement_data = monitoring_data.get('engagement_metrics', {})
    success_reports = monitoring_data.get('success_reports', 0)

    # Handle case where data might be structured differently
    if not mention_data and not engagement_data and not success_reports:
        # Check alternative locations
        interactions = vulnerability_data.get('interactions', {})
        aggregations = vulnerability_data.get('aggregations', {})
        
        if interactions:
            mention_data = {'platform': len(interactions)}
            engagement_data = interactions
        elif aggregations:
            mention_data = {'platform': aggregations.get('interaction_counts', {}).get('total', 0)}

    mention_score = _calculate_mention_score(mention_data)
    engagement_score = _calculate_engagement_score(engagement_data)
    validation_score = _calculate_validation_score(success_reports)
    platform_score = min(len(mention_data) * 15, 100) if mention_data else 0

    adoption_score = (
        0.3 * mention_score +
        0.3 * engagement_score +
        0.25 * validation_score +
        0.15 * platform_score
    )

    adoption_score = round(adoption_score)

    evidence = {
        'mention_assessment': mention_score,
        'engagement_assessment': engagement_score,
        'validation_assessment': validation_score,
        'platform_spread': platform_score,
        'justification': _generate_adoption_justification(adoption_score)
    }

    return adoption_score, evidence


def _calculate_mention_score(mention_data):
    """
    Calculate score based on mention volume and distribution.
    """
    if not mention_data:
        return 0
        
    total_mentions = sum(mention_data.values()) if isinstance(mention_data, dict) else 0
    
    # Logarithmic scaling to avoid over-weighting very high mention counts
    # 1 mention = 10, 10 mentions = 40, 100 mentions = 70, 1000+ mentions = 100
    if total_mentions == 0:
        return 0
    elif total_mentions == 1:
        return 10
    else:
        return min(10 + 20 * math.log10(total_mentions), 100)

def _calculate_engagement_score(engagement_data):
    if not isinstance(engagement_data, dict):
        return 0

    upvotes = engagement_data.get("upvotes", 0)
    comments = engagement_data.get("comments", 0)
    shares = engagement_data.get("shares", 0)
    messages = engagement_data.get("messages", 0)
    replies = engagement_data.get("replies", 0)
    reactions = engagement_data.get("reactions", 0)

    total_engagement = upvotes + comments + (2 * shares) + messages + replies + reactions

    # Normalize to a 0-100 scale
    if total_engagement == 0:
        return 0
    elif total_engagement < 10:
        return 20
    elif total_engagement < 50:
        return 40
    elif total_engagement < 200:
        return 60
    elif total_engagement < 500:
        return 80
    else:
        return 100


def _calculate_validation_score(success_reports):
    if not isinstance(success_reports, (int, float)):
        return 0
    success_reports = int(success_reports)
    if success_reports == 0:
        return 0
    elif success_reports < 5:
        return 20
    elif success_reports < 15:
        return 50
    elif success_reports < 30:
        return 75
    else:
        return 100


def _generate_adoption_justification(score):
    return f"Final adoption score: {score}"