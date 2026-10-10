"""
Independent oracle test for scale factor skills.
Derives answers purely from rendered question text, without reading the generator.
"""

import re
import math
from fractions import Fraction
import pytest
import mathgen


class TestScaleFactorOracle:
    """Oracle verification of scale factor questions."""

    @staticmethod
    def parse_easy_similar_areas(text: str) -> dict:
        """Parse easy scale_factor_similar_areas question.
        Pattern: "The scale factor from Figure A to Figure B is p/q.
        Figure A has an area of B cm²."
        """
        scale_match = re.search(r"scale factor from Figure A to Figure B is (\d+)/(\d+)", text)
        area_match = re.search(r"Figure A has an area of (\d+)\s*cm[²2]", text)

        assert scale_match and area_match, f"Cannot parse easy areas: {text}"
        p, q = int(scale_match.group(1)), int(scale_match.group(2))
        area_a = int(area_match.group(1))

        return {"p": p, "q": q, "area_a": area_a}

    @staticmethod
    def parse_medium_similar_areas(text: str) -> dict:
        """Parse medium scale_factor_similar_areas question.
        Pattern: "A side of Figure A measures L1 cm; the corresponding side
        of Figure B measures L2 cm. Figure A has an area of B cm²."
        """
        # Extract both side lengths
        sides_match = re.search(
            r"A side of Figure A measures (\d+) cm;.*?corresponding side of Figure B measures (\d+) cm", text
        )
        area_match = re.search(r"Figure A has an area of (\d+)\s*cm[²2]", text)

        assert sides_match and area_match, f"Cannot parse medium areas: {text}"
        side_a = int(sides_match.group(1))
        side_b = int(sides_match.group(2))
        area_a = int(area_match.group(1))

        return {"side_a": side_a, "side_b": side_b, "area_a": area_a}

    @staticmethod
    def parse_hard_similar_areas(text: str) -> dict:
        """Parse hard scale_factor_similar_areas question.
        Pattern: "Figure A has an area of A1 cm² and Figure B has an area of A2 cm².
        A side of Figure A measures L1 cm."
        """
        areas_match = re.search(
            r"Figure A has an area of (\d+)\s*cm[²2].*?Figure B has an area of (\d+)\s*cm[²2]", text
        )
        length_match = re.search(r"A side of Figure A measures (\d+) cm", text)

        assert areas_match and length_match, f"Cannot parse hard areas: {text}"
        area_a = int(areas_match.group(1))
        area_b = int(areas_match.group(2))
        length_a = int(length_match.group(1))

        return {"area_a": area_a, "area_b": area_b, "length_a": length_a}

    @staticmethod
    def parse_easy_similar_solids(text: str) -> dict:
        """Parse easy scale_factor_similar_solids question.
        Pattern: "The scale factor from Solid A to Solid B is p/q.
        Solid A has a surface area of B cm² / volume of V cm³."
        """
        scale_match = re.search(r"scale factor from Solid A to Solid B is (\d+)/(\d+)", text)
        surface_match = re.search(r"Solid A has a surface area of (\d+)\s*cm[²2]", text)
        volume_match = re.search(r"Solid A has a volume of (\d+)\s*cm[³3]", text)

        assert scale_match, f"Cannot parse easy solids scale factor: {text}"
        p, q = int(scale_match.group(1)), int(scale_match.group(2))

        # Determine if it's surface area or volume
        if surface_match:
            value = int(surface_match.group(1))
            quantity_type = "surface_area"
        elif volume_match:
            value = int(volume_match.group(1))
            quantity_type = "volume"
        else:
            raise AssertionError(f"Cannot find quantity type in: {text}")

        return {"p": p, "q": q, "value": value, "quantity_type": quantity_type}

    @staticmethod
    def parse_medium_similar_solids(text: str) -> dict:
        """Parse medium scale_factor_similar_solids question.
        Pattern: "A side of Solid A measures L1 cm; the corresponding side
        of Solid B measures L2 cm. Solid A has a volume/surface area of V."
        """
        sides_match = re.search(
            r"A side of Solid A measures (\d+) cm;.*?corresponding side of Solid B measures (\d+) cm", text
        )
        volume_match = re.search(r"Solid A has a volume of (\d+)\s*cm[³3]", text)
        surface_match = re.search(r"Solid A has a surface area of (\d+)\s*cm[²2]", text)

        assert sides_match, f"Cannot parse medium solids sides: {text}"
        side_a = int(sides_match.group(1))
        side_b = int(sides_match.group(2))

        if volume_match:
            value = int(volume_match.group(1))
            quantity_type = "volume"
        elif surface_match:
            value = int(surface_match.group(1))
            quantity_type = "surface_area"
        else:
            raise AssertionError(f"Cannot find quantity type in: {text}")

        return {"side_a": side_a, "side_b": side_b, "value": value, "quantity_type": quantity_type}

    @staticmethod
    def parse_hard_similar_solids(text: str) -> dict:
        """Parse hard scale_factor_similar_solids question.
        Pattern A: "Solid A has a volume of V1 cm³ and Solid B has a volume of V2 cm³.
                    Solid A has a surface area of SA1 cm²."
        Pattern B: "Solid A has a surface area of SA1 cm² and Solid B has a surface area of SA2 cm².
                    Solid A has a volume of V1 cm³."
        """
        volumes_match = re.search(
            r"Solid A has a volume of (\d+)\s*cm[³3].*?Solid B has a volume of (\d+)\s*cm[³3]", text
        )
        areas_match = re.search(
            r"Solid A has a surface area of (\d+)\s*cm[²2].*?Solid B has a surface area of (\d+)\s*cm[²2]", text
        )

        # Pattern A: volumes given, asking for surface area
        if volumes_match:
            # Check if we're asking for surface area by looking at what comes after the question mark
            if "What is Solid B's surface area" in text:
                volume_a = int(volumes_match.group(1))
                volume_b = int(volumes_match.group(2))
                surface_match = re.search(r"Solid A has a surface area of (\d+)\s*cm[²2]", text)
                assert surface_match, f"Cannot find Solid A surface area: {text}"
                surface_a = int(surface_match.group(1))
                return {
                    "known_pair_type": "volume",
                    "known_a": volume_a,
                    "known_b": volume_b,
                    "value": surface_a,
                    "solve_for": "surface_area",
                }

        # Pattern B: surface areas given, asking for volume
        if areas_match:
            # Check if we're asking for volume
            if "What is Solid B's volume" in text:
                surface_a = int(areas_match.group(1))
                surface_b = int(areas_match.group(2))
                volume_match = re.search(r"Solid A has a volume of (\d+)\s*cm[³3]", text)
                assert volume_match, f"Cannot find Solid A volume: {text}"
                volume_a = int(volume_match.group(1))
                return {
                    "known_pair_type": "surface_area",
                    "known_a": surface_a,
                    "known_b": surface_b,
                    "value": volume_a,
                    "solve_for": "volume",
                }

        raise AssertionError(f"Cannot parse hard solids: {text}")

    @staticmethod
    def compute_easy_areas(p: int, q: int, area_a: int) -> Fraction:
        """Compute area of figure B given scale factor and area A."""
        scale = Fraction(p, q)
        return Fraction(area_a) * scale * scale

    @staticmethod
    def compute_medium_areas(side_a: int, side_b: int, area_a: int) -> Fraction:
        """Compute area of figure B given corresponding side lengths and area A."""
        scale = Fraction(side_b, side_a)
        return Fraction(area_a) * scale * scale

    @staticmethod
    def compute_hard_areas(area_a: int, area_b: int, length_a: int) -> Fraction:
        """Compute length of figure B given both areas and one length of A.
        Linear scale factor is sqrt(area_b / area_a).
        length_b = length_a * sqrt(area_b / area_a)
        """
        area_ratio = Fraction(area_b, area_a)
        # Compute sqrt(area_b / area_a) using exact integer square root
        # area_b / area_a after reducing
        num_sqrt = math.isqrt(area_ratio.numerator)
        den_sqrt = math.isqrt(area_ratio.denominator)

        # Verify it's a perfect square
        if num_sqrt * num_sqrt != area_ratio.numerator or den_sqrt * den_sqrt != area_ratio.denominator:
            raise ValueError(f"Not a perfect square: {area_ratio}")

        scale = Fraction(num_sqrt, den_sqrt)
        return Fraction(length_a) * scale

    @staticmethod
    def compute_easy_solids(p: int, q: int, value: int, quantity_type: str) -> Fraction:
        """Compute quantity of solid B given scale factor and quantity of A."""
        scale = Fraction(p, q)
        if quantity_type == "surface_area":
            return Fraction(value) * scale * scale
        else:  # volume
            return Fraction(value) * scale * scale * scale

    @staticmethod
    def compute_medium_solids(side_a: int, side_b: int, value: int, quantity_type: str) -> Fraction:
        """Compute quantity of solid B given corresponding side lengths and quantity of A."""
        scale = Fraction(side_b, side_a)
        if quantity_type == "surface_area":
            return Fraction(value) * scale * scale
        else:  # volume
            return Fraction(value) * scale * scale * scale

    @staticmethod
    def _cbrt(n: int) -> int:
        """Compute integer cube root using binary search."""
        if n == 0:
            return 0
        # Binary search for cube root
        low, high = 0, max(1, n)
        while low <= high:
            mid = (low + high) // 2
            cube = mid**3
            if cube == n:
                return mid
            elif cube < n:
                low = mid + 1
            else:
                high = mid - 1
        raise ValueError(f"{n} is not a perfect cube")

    @staticmethod
    def compute_hard_solids(known_pair_type: str, known_a: int, known_b: int, value: int, solve_for: str) -> Fraction:
        """Compute the unknown quantity for hard similar solids.

        If known_pair_type is 'volume':
            Linear scale factor = cbrt(known_b / known_a)
            If solving for surface_area: value * (known_b / known_a)^(2/3)

        If known_pair_type is 'surface_area':
            Linear scale factor = sqrt(known_b / known_a)
            If solving for volume: value * (known_b / known_a)^(3/2)
        """
        if known_pair_type == "volume":
            # Linear scale factor cubed = known_b / known_a
            volume_ratio = Fraction(known_b, known_a)
            # Compute cbrt(volume_ratio) using exact cube root
            num_cbrt = TestScaleFactorOracle._cbrt(volume_ratio.numerator)
            den_cbrt = TestScaleFactorOracle._cbrt(volume_ratio.denominator)

            linear_scale = Fraction(num_cbrt, den_cbrt)
            if solve_for == "surface_area":
                return Fraction(value) * linear_scale * linear_scale
            else:
                raise ValueError(f"Unexpected solve_for with volume pair: {solve_for}")

        else:  # known_pair_type == 'surface_area'
            # Linear scale factor squared = known_b / known_a
            area_ratio = Fraction(known_b, known_a)
            # Compute sqrt(area_ratio)
            num_sqrt = math.isqrt(area_ratio.numerator)
            den_sqrt = math.isqrt(area_ratio.denominator)

            # Verify it's a perfect square
            if num_sqrt * num_sqrt != area_ratio.numerator or den_sqrt * den_sqrt != area_ratio.denominator:
                raise ValueError(f"Not a perfect square: {area_ratio}")

            linear_scale = Fraction(num_sqrt, den_sqrt)
            if solve_for == "volume":
                return Fraction(value) * linear_scale * linear_scale * linear_scale
            else:
                raise ValueError(f"Unexpected solve_for with surface area pair: {solve_for}")

    @pytest.mark.parametrize("topic", ["scale_factor_similar_areas", "scale_factor_similar_solids"])
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", range(60))
    def test_scale_factor_oracle(self, topic: str, difficulty: str, seed: int):
        """Test that independently derived answer matches the generated answer."""

        # Generate the question
        question = mathgen.make(topic, seed=seed, difficulty=difficulty)

        # Parse based on topic and difficulty
        if topic == "scale_factor_similar_areas":
            if difficulty == "easy":
                parsed = self.parse_easy_similar_areas(question.text)
                expected = self.compute_easy_areas(parsed["p"], parsed["q"], parsed["area_a"])
            elif difficulty == "medium":
                parsed = self.parse_medium_similar_areas(question.text)
                expected = self.compute_medium_areas(parsed["side_a"], parsed["side_b"], parsed["area_a"])
            else:  # hard
                parsed = self.parse_hard_similar_areas(question.text)
                expected = self.compute_hard_areas(parsed["area_a"], parsed["area_b"], parsed["length_a"])

        else:  # scale_factor_similar_solids
            if difficulty == "easy":
                parsed = self.parse_easy_similar_solids(question.text)
                expected = self.compute_easy_solids(parsed["p"], parsed["q"], parsed["value"], parsed["quantity_type"])
            elif difficulty == "medium":
                parsed = self.parse_medium_similar_solids(question.text)
                expected = self.compute_medium_solids(
                    parsed["side_a"], parsed["side_b"], parsed["value"], parsed["quantity_type"]
                )
            else:  # hard
                parsed = self.parse_hard_similar_solids(question.text)
                expected = self.compute_hard_solids(
                    parsed["known_pair_type"],
                    parsed["known_a"],
                    parsed["known_b"],
                    parsed["value"],
                    parsed["solve_for"],
                )

        # Assert answer matches
        actual = Fraction(question.answer_text)
        assert expected == actual, (
            f"Mismatch for {topic} {difficulty} seed {seed}: "
            f"expected {expected}, got {actual}\n"
            f"Question: {question.text}"
        )

        # Assert all distractors are different from expected answer
        for distractor_text in question.distractors:
            distractor = Fraction(distractor_text)
            assert distractor != expected, (
                f"Distractor matches expected for {topic} {difficulty} seed {seed}: "
                f"distractor={distractor}, expected={expected}\n"
                f"Question: {question.text}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
