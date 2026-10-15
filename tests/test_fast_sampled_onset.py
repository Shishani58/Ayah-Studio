"""Fractional forecast crops must retain their actual acoustic time origin."""
from dataclasses import replace

from test_fast_executor import Fake, observation, pause, run


def test_wasl_onset_uses_sampled_crop_start_and_passes_strict_validator():
    def respond(fake, start, duration):
        if start == 1000:
            return observation(fake.phrase(1))
        if start == 6000:
            return observation(fake.phrase(2))
        if start > 6100 and duration >= 5000:
            heard = observation((fake.rendered[5][0], *fake.phrase(3)))
            # For these fixed fake weights, the crop rounds from21835.2941
            # to21835.3125ms. This relative offset straddles an integer half.
            return replace(heard, positions_ms=(0., 2248.2, 2448.2))
        return observation(())

    fake = Fake(respond)
    units = list(fake.rendered)
    units[0] = (units[0][0] + "x",)
    units[4] = ("a2w4" + "x" * 27,)
    units[5] = ("a2w5" + "x" * 27,)
    fake.rendered = tuple(units)
    rows, trace = run(fake, [pause(1000), pause(6000)])
    boundary = trace["boundaries"][2]
    assert boundary["evidence_kind"] == "wasl" and rows[2][2] > 0
    read = trace["acoustic_reads"][boundary["read_id"]]
    relative = read["observed"]["positions_ms"][boundary["proof"]["offset"]]
    actual = round(read["window_ms"][0] + relative)
    requested = round(read["requested_window_ms"][0] + relative)
    assert actual != requested
    assert rows[2][1] == actual
