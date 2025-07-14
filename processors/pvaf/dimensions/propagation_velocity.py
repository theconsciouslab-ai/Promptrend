from datetime import datetime, timezone

"""
Dimension: Propagation Velocity

Measures how quickly the vulnerability spreads across platforms and communities.
"""

def calculate(vulnerability_data):
    # Try standard location
    monitoring_data = vulnerability_data.get("monitoring_data")

    # Fallback: adapt from metadata.temporal_data 
    if monitoring_data is None:
        temporal_data = vulnerability_data.get("metadata", {}).get("temporal_data", {})
        platform_timestamps = {}
        for platform, timestamp_str in temporal_data.items():
            try:
                dt = _parse_datetime(timestamp_str)
                if dt:
                    platform_timestamps[platform] = dt
            except Exception:
                continue

        if platform_timestamps:
            first_observed = min(platform_timestamps.values())
            monitoring_data = {
                "first_observed": first_observed,
                "platform_timestamps": platform_timestamps,
                "mention_timeline": {}  # optional
            }
        else:
            # Fallback to created_at and collection_time
            created_at = vulnerability_data.get("created_at")
            collection_time = vulnerability_data.get("collection_time")
            
            timestamps = {}
            if created_at:
                dt = _parse_datetime(created_at)
                if dt:
                    timestamps["discovery"] = dt
            
            if collection_time:
                dt = _parse_datetime(collection_time)
                if dt:
                    timestamps["collection"] = dt
            
            # Extract platform from platform_data
            platform = vulnerability_data.get("platform", vulnerability_data.get("platform_data", {}).get("platform", "unknown"))
            if timestamps:
                first_observed = min(timestamps.values())
                platform_timestamps = {platform: first_observed}
                monitoring_data = {
                    "first_observed": first_observed,
                    "platform_timestamps": platform_timestamps,
                    "mention_timeline": {}
                }
            else:
                monitoring_data = {}

    return calculate_propagation_velocity(monitoring_data)


def _parse_datetime(dt_input):
    """
    Parse datetime from various formats and ensure timezone consistency.
    Always returns timezone-aware datetime in UTC or None if parsing fails.
    """
    if dt_input is None:
        return None
    
    if isinstance(dt_input, datetime):
        # If timezone-naive, assume UTC
        if dt_input.tzinfo is None:
            return dt_input.replace(tzinfo=timezone.utc)
        # If timezone-aware, convert to UTC
        return dt_input.astimezone(timezone.utc)
    
    if isinstance(dt_input, str):
        try:
            # Handle various string formats
            dt_str = dt_input.strip()
            
            # Remove 'Z' and replace with explicit UTC
            if dt_str.endswith('Z'):
                dt_str = dt_str[:-1] + '+00:00'
            
            # Parse the datetime
            try:
                dt = datetime.fromisoformat(dt_str)
            except ValueError:
                # Try other common formats
                for fmt in [
                    '%Y-%m-%d %H:%M:%S.%f',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S.%f',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d'
                ]:
                    try:
                        dt = datetime.strptime(dt_str.split('+')[0].split('Z')[0], fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return None
            
            # Ensure timezone awareness
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            
            return dt
            
        except Exception:
            return None
    
    return None


def calculate_propagation_velocity(monitoring_data):
    """
    Calculate Propagation Velocity score (0-100) based on monitoring data.
    
    Parameters:
    - monitoring_data: Dict containing temporal spread data:
        - first_observed: Timestamp of first observation
        - platform_timestamps: Dict mapping platforms to first observation
        - mention_timeline: Dict mapping dates to mention counts
        
    Returns:
    - velocity_score: Integer score from 0-100
    - evidence: Dict with justification for the score
    """
    # Get timestamps for initial discovery and subsequent platforms
    first_observed = monitoring_data.get('first_observed')
    platform_timestamps = monitoring_data.get('platform_timestamps', {})
    mention_timeline = monitoring_data.get('mention_timeline', {})

    # Guard against missing or invalid data
    if not first_observed and not platform_timestamps:
        return 0, {'error': 'Missing required propagation data'}

    # Parse first_observed
    first_observed = _parse_datetime(first_observed)

    # Parse all platform timestamps and ensure consistency
    parsed_timestamps = {}
    for platform, ts in platform_timestamps.items():
        parsed_ts = _parse_datetime(ts)
        if parsed_ts:
            parsed_timestamps[platform] = parsed_ts

    # If no first_observed but have platform_timestamps, use the earliest
    if not first_observed and parsed_timestamps:
        first_observed = min(parsed_timestamps.values())

    if not first_observed:
        return 5, {'error': 'Cannot determine first observation time', 'default_score_applied': True}

    if len(parsed_timestamps) <= 1:
        cross_platform_score = 5  # Single platform gets minimal score
    else:
        try:
            datetime_stamps = list(parsed_timestamps.values())
            
            if len(datetime_stamps) < 2:
                cross_platform_score = 5
            else:
                sorted_timestamps = sorted(datetime_stamps)
                
                # Calculate time difference in hours
                time_diff = sorted_timestamps[1] - first_observed
                hours_to_second = time_diff.total_seconds() / 3600

                if hours_to_second < 6:
                    cross_platform_score = 80
                elif hours_to_second < 24:
                    cross_platform_score = 70
                elif hours_to_second < 48:
                    cross_platform_score = 50
                elif hours_to_second < 96:
                    cross_platform_score = 30
                else:
                    cross_platform_score = 20

                if len(datetime_stamps) >= 4:
                    time_diff_fourth = sorted_timestamps[3] - first_observed
                    hours_to_fourth = time_diff_fourth.total_seconds() / 3600
                    if hours_to_fourth < 48:
                        cross_platform_score += 15

        except Exception as e:
            cross_platform_score = 5  # Fallback if time calc fails

    mention_acceleration = _calculate_mention_acceleration(mention_timeline)
    sharing_score = _calculate_sharing_velocity(mention_timeline)

    velocity_score = (
        0.5 * cross_platform_score +
        0.3 * sharing_score +
        0.2 * mention_acceleration
    )

    evidence = {
        'cross_platform_time': cross_platform_score,
        'sharing_velocity': sharing_score,
        'mention_acceleration': mention_acceleration,
        'platform_spread': len(parsed_timestamps),
        'first_observed': first_observed.isoformat() if first_observed else None,
        'justification': _generate_velocity_justification(velocity_score)
    }

    return round(velocity_score), evidence


def _calculate_mention_acceleration(mention_timeline):
    if not mention_timeline:
        return 0
    days = sorted(mention_timeline.keys())
    if len(days) < 2:
        return 10
    return min(len(days) * 5, 100)


def _calculate_sharing_velocity(mention_timeline):
    if not mention_timeline:
        return 0
    total_mentions = sum(mention_timeline.values())
    if total_mentions < 5:
        return 10
    elif total_mentions < 20:
        return 30
    elif total_mentions < 100:
        return 60
    else:
        return 90


def _generate_velocity_justification(score):
    return f"Velocity score: {score}"