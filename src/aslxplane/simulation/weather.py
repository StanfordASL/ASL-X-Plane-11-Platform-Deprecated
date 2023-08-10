from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import random

import numpy as np

try:
    from aslxplane.utils.robust_xpc import RobustXPlaneConnect
except ImportError:
    root_path = Path(__file__).absolute().parents[2]
    if str(root_path) not in sys.path:
        sys.path.append(str(root_path))


WEATHER_STATES = {
    "cloud_bottom1": "sim/weather/cloud_base_msl_m[0]",
    "cloud_bottom2": "sim/weather/cloud_base_msl_m[1]",
    "cloud_bottom3": "sim/weather/cloud_base_msl_m[2]",
    "cloud_top1": "sim/weather/cloud_tops_msl_m[0]",
    "cloud_top2": "sim/weather/cloud_tops_msl_m[1]",
    "cloud_top3": "sim/weather/cloud_tops_msl_m[2]",
    "cloud_coverage1": "sim/weather/cloud_coverage[0]",
    "cloud_coverage2": "sim/weather/cloud_coverage[1]",
    "cloud_coverage3": "sim/weather/cloud_coverage[2]",
    "cloud_type1": "sim/weather/cloud_type[0]",
    "cloud_type2": "sim/weather/cloud_type[1]",
    "cloud_type3": "sim/weather/cloud_type[2]",
    "rain": "sim/weather/rain_percent",
    "temperature": "sim/weather/temperature_sealevel_c",
}

DAY_OF_YEAR = "sim/time/local_date_days"

utc_m8_hour_to_sec = lambda h: (((h + 8) * 3600) % (60 * 60 * 24))


def randomize_the_weather(
    clouds: str = None, rain_snow: str = None, time_of_day: str = None
) -> dict[str, Any]:
    xp = randomize_the_weather.xp

    xp.sendDREF(DAY_OF_YEAR, 293)  # should be October 21st or so, equal day and night
    # assume UTC-8, dark is 6pm to 6am

    time_of_day = (
        random.choice(["day", "night", "dusk", "dawn"]) if time_of_day is None else time_of_day
    )
    if time_of_day == "day":
        time_since_midnight = utc_m8_hour_to_sec(random.uniform(7, 17))
    elif time_of_day == "night":
        time_since_midnight = utc_m8_hour_to_sec(random.uniform(19, 29))
    elif time_of_day == "dawn":
        time_since_midnight = utc_m8_hour_to_sec(random.uniform(6.4, 7.1))
    elif time_of_day == "dusk":
        time_since_midnight = utc_m8_hour_to_sec(random.uniform(17.5, 17.9))
    xp.sendDREF("sim/time/zulu_time_sec", time_since_midnight)

    clouds = random.choice(["clouds", "none"]) if clouds is None else clouds
    if clouds == "none":
        cloud_cover = [0.0, 0.0, 0.0]
        cloud_type = [0.0, 0.0, 0.0]
        cloud_heights = [0.0, 0.0, 0.0]
        xp.sendDREFs([WEATHER_STATES[f"cloud_coverage{i}"] for i in range(1, 4)], cloud_cover)
        xp.sendDREFs([WEATHER_STATES[f"cloud_type{i}"] for i in range(1, 4)], cloud_type)
        rain_snow_none = "none" if rain_snow is None else rain_snow
    else:
        cloud_cover = [random.uniform(1.0, 4.0) for _ in range(3)]
        cloud_heights = np.cumsum(np.random.rand(3) * 1000.0)
        cloud_type = [random.choice([1, 2, 3, 4, 5]) for _ in range(3)]
        xp.sendDREFs([WEATHER_STATES[f"cloud_coverage{i}"] for i in range(1, 4)], cloud_cover)
        xp.sendDREFs([WEATHER_STATES[f"cloud_type{i}"] for i in range(1, 4)], cloud_type)
        xp.sendDREFs([WEATHER_STATES[f"cloud_bottom{i}"] for i in range(1, 4)], cloud_heights)
        xp.sendDREFs([WEATHER_STATES[f"cloud_top{i}"] for i in range(1, 4)], cloud_heights + 2e3)
        rain_snow_none = random.choice(["rain", "snow", "none"]) if rain_snow is None else rain_snow

    if rain_snow_none == "none":
        rain_percent = 0.0
        xp.sendDREF(WEATHER_STATES["rain"], 0.0)
    elif rain_snow_none == "rain":
        rain_percent = random.uniform(0.0, 1.0)
        xp.sendDREF(WEATHER_STATES["rain"], rain_percent)
        xp.sendDREF(WEATHER_STATES["temperature"], random.uniform(5.0, 40.0))
    elif rain_snow_none == "snow":
        rain_percent = random.uniform(0.0, 1.0)
        xp.sendDREF(WEATHER_STATES["rain"], rain_percent)
        xp.sendDREF(WEATHER_STATES["temperature"], random.uniform(-20.0, -5.0))

    return dict(
        time_of_day=time_of_day,
        time_since_midnight=time_since_midnight,
        cloud_cover=cloud_cover,
        rain_snow_none=rain_snow_none,
        cloud_type=cloud_type,
        cloud_heights=cloud_heights,
        rain_percent=rain_percent,
    )


randomize_the_weather.xp = RobustXPlaneConnect()
