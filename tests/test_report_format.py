"""Checks of the running and cycling reports on made-up measures (no video needed).

Run with: python -m tests.test_report_format
"""

import numpy as np

from src.sports import cycling, running


def titles(sections):
    return [title for title, _ in sections]


def text(sections):
    return "\n".join(line for _, lines in sections for line in lines)


def fake_leg(flex=20, peak=40, landing=5, strikes=20):
    rng = np.random.default_rng(1)
    return {
        "strikes_used": strikes,
        "flex_touchdown": list(flex + rng.normal(0, 1, strikes)),
        "peak_flexion": list(peak + rng.normal(0, 1, strikes)),
        "landing": list(landing + rng.normal(0, 1, strikes)),
        "tibia": list(5 + rng.normal(0, 1, strikes)),
        "sensitivity": [(1, landing, 5), (2, landing - 2, 5), (3, landing - 4, 5)],
    }


def check_running():
    osc = list(np.random.default_rng(2).normal(7, 0.3, 30))
    trunk = list(np.random.default_rng(3).normal(6, 1, 200))
    legs = {"left": fake_leg(), "right": fake_leg()}
    main, bg = running.build_report(240, 20, 14, 171, osc, trunk, legs, "treadmill")
    assert titles(main)[:2] == ["YOUR RESULT", "WHAT TO DO"], titles(main)
    assert "CHECK IT YOURSELF" in titles(main)
    assert "Nothing in this video clearly breaks" in text(main)
    body = text(main)
    # Speed 14 km/h: reference about 176 steps/min; 171 is within one SD, so only an experiment
    assert "(5% more than your 171" in body and "Your steps are on the slow side" not in body
    assert "Let your arms swing freely" in body and "Land softly and quietly" in body
    assert "regular runners in a lab study at about 14 km/h" in body
    assert "SOURCES" in titles(bg) and "KNEE (flexion: 0° = straight leg)" in titles(bg)
    # Different legs give a coach's tip
    legs = {"left": fake_leg(peak=30), "right": fake_leg(peak=50)}
    main, _ = running.build_report(240, 20, 14, 171, osc, trunk, legs, "treadmill")
    assert "Your two legs do not move the same way" in text(main)
    assert "coach's tip" in text(main)
    # Slow steps for the speed: coach's tip with the personalised number
    main, _ = running.build_report(240, 20, 14, 150, osc, trunk,
                                   {"left": fake_leg(), "right": fake_leg()}, "treadmill")
    assert "Your steps are on the slow side for your speed" in text(main)
    assert "about 158 steps per minute" in text(main)
    # No speed: no reference and no slow-steps tip
    main, _ = running.build_report(240, 20, None, 150, osc, trunk,
                                   {"left": fake_leg(), "right": fake_leg()}, "treadmill")
    assert "slow side" not in text(main) and "regular runners in a lab" not in text(main)
    # Few strikes per leg: the leg is not summarised
    main, _ = running.build_report(240, 20, 14, 171, osc, trunk,
                                   {"left": fake_leg(strikes=9), "right": fake_leg(strikes=9)},
                                   "treadmill")
    assert "not summarized" in text(main)
    # Reference cadence by speed
    assert running.reference_cadence(None) is None and running.reference_cadence(20) is None
    assert abs(running.reference_cadence(12.6)[0] - 171.4) < 0.01
    # Low frame rate: filming tips appear
    main, _ = running.build_report(30, 20, None, float("nan"), osc, trunk, legs, "treadmill")
    assert "FILMING TIPS" in titles(main)
    assert "Your two legs do not move the same way" not in text(main)   # too unreliable at 30 fps


def fake_measures(knee):
    rng = np.random.default_rng(4)
    n = 30

    def v(m, s=1.0):
        return list(m + rng.normal(0, s, n))
    return {"rpm": v(80), "flex_bdc": v(knee), "max_ext": v(knee - 2), "knee_range": v(70),
            "ankle_bdc": v(110), "ankle_range": v(25), "hip_min": v(60)}


def check_cycling():
    rng = np.random.default_rng(5)
    trunk, reach, elbow = (list(rng.normal(m, 2, 100)) for m in (40, 60, 160))
    # Knee 54 degrees: saddle probably too low
    main, bg = cycling.build_report(120, 30, fake_measures(54), trunk, reach, elbow)
    body = text(main)
    assert titles(main)[:2] == ["YOUR RESULT", "WHAT TO DO"], titles(main)
    assert "probably too low" in body and "raising" in body and "backed by research" in body
    assert "CHECK IT YOURSELF" in titles(main) and "Stay active" in body
    assert "SOURCES" in titles(bg)
    # Knee 35: fine, no tip
    main, _ = cycling.build_report(120, 30, fake_measures(35), trunk, reach, elbow)
    assert "saddle height looks fine" in text(main).lower()
    assert "probably too" not in text(main)
    # Knee 8: too high
    main, _ = cycling.build_report(120, 30, fake_measures(8), trunk, reach, elbow)
    assert "probably too high" in text(main) and "lowering" in text(main)
    # Few revolutions: cannot judge
    short = {k: v[:3] for k, v in fake_measures(54).items()}
    main, _ = cycling.build_report(120, 30, short, trunk, reach, elbow)
    assert "cannot judge" in text(main) and "probably too" not in text(main)


if __name__ == "__main__":
    check_running()
    check_cycling()
    print("All report checks passed.")