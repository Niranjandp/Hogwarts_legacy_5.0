import unittest

from core import EV, optimize_charging_plan, validate_solution
from data.station_finder import get_stations_along_route


class DynamicProgrammingOptimizerTests(unittest.TestCase):
    def test_default_route_returns_valid_plan_without_mutating_ev(self):
        ev = EV()
        original_soc = ev.current_soc_percent
        stations = get_stations_along_route()

        plan = optimize_charging_plan(ev, stations)

        self.assertTrue(plan.is_feasible, plan.violations)
        self.assertTrue(plan.stop_sequence)
        self.assertGreaterEqual(plan.battery_profile[-1]["soc"], 15.0)
        self.assertEqual(ev.current_soc_percent, original_soc)
        validated, violations, _ = validate_solution(
            ev, stations, plan.stop_sequence, plan.charge_amounts
        )
        self.assertTrue(validated, violations)

    def test_unreachable_route_reports_infeasibility(self):
        plan = optimize_charging_plan(EV(), [])

        self.assertFalse(plan.is_feasible)
        self.assertTrue(plan.violations)


if __name__ == "__main__":
    unittest.main()