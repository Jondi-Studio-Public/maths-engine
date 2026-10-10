"""
Independent oracle test for pyramid and faces surface area skills.
Derives answers purely from rendered question text, without reading the generator.
"""

import re
from fractions import Fraction
import pytest
import mathgen


class TestPyramidFacesOracle:
    """Oracle verification of pyramid and faces questions."""

    @staticmethod
    def parse_pyramid_square(text: str) -> Fraction:
        """Parse square pyramid: base area + 4 lateral triangles."""
        base_match = re.search(r"square base with side length (\d+)\s*cm", text)
        slant_match = re.search(r"Each of its four triangular faces has a slant height of (\d+)\s*cm", text)

        assert base_match and slant_match, f"Cannot parse square pyramid: {text}"

        s = Fraction(base_match.group(1))
        l = Fraction(slant_match.group(1))

        base_area = s * s
        lateral_area = 4 * (s * l / 2)
        return base_area + lateral_area

    @staticmethod
    def parse_pyramid_rectangular(text: str) -> Fraction:
        """Parse rectangular pyramid: base area + lateral triangles."""
        base_match = re.search(r"rectangular base measuring (\d+)\s*cm by (\d+)\s*cm", text)
        longer_match = re.search(r"longer sides have a slant height of (\d+)\s*cm", text)
        shorter_match = re.search(r"shorter sides have a slant height of (\d+)\s*cm", text)

        assert base_match and longer_match and shorter_match, f"Cannot parse rectangular pyramid: {text}"

        a = Fraction(base_match.group(1))
        b = Fraction(base_match.group(2))
        l_long = Fraction(longer_match.group(1))
        l_short = Fraction(shorter_match.group(1))

        # Determine which is longer/shorter
        longer = max(a, b)
        shorter = min(a, b)

        base_area = a * b
        lateral_area = 2 * (longer * l_long / 2) + 2 * (shorter * l_short / 2)
        return base_area + lateral_area

    @staticmethod
    def parse_pyramid_right_triangle(text: str) -> Fraction:
        """Parse right-triangular pyramid: base area + 3 lateral triangles."""
        base_match = re.search(
            r"right-triangular base with legs (\d+)\s*cm and (\d+)\s*cm and hypotenuse (\d+)\s*cm", text
        )

        assert base_match, f"Cannot parse right-triangular base: {text}"

        leg1 = Fraction(base_match.group(1))
        leg2 = Fraction(base_match.group(2))
        hyp = Fraction(base_match.group(3))

        # Extract slant heights for each edge
        edge_slants = {}
        edge_patterns = [
            (leg1, r"face on the {} cm edge has a slant height of (\d+)\s*cm"),
            (leg2, r"face on the {} cm edge has a slant height of (\d+)\s*cm"),
            (hyp, r"face on the {} cm edge has a slant height of (\d+)\s*cm"),
        ]

        for edge, pattern_template in edge_patterns:
            pattern = pattern_template.format(edge)
            match = re.search(pattern, text)
            assert match, f"Cannot parse slant height for edge {edge}: {text}"
            edge_slants[edge] = Fraction(match.group(1))

        base_area = leg1 * leg2 / 2
        lateral_area = (leg1 * edge_slants[leg1] / 2) + (leg2 * edge_slants[leg2] / 2) + (hyp * edge_slants[hyp] / 2)
        return base_area + lateral_area

    @staticmethod
    def parse_faces(text: str) -> Fraction:
        """Parse addition of faces: parse all Face N sentences and excluded face."""
        total_area = Fraction(0)
        parsed_faces = []

        # Parse rectangles: "Face N is a rectangle A cm by B cm."
        rect_pattern = r"Face (\d+) is a rectangle (\d+)\s*cm by (\d+)\s*cm"
        for match in re.finditer(rect_pattern, text):
            face_num = int(match.group(1))
            a = Fraction(match.group(2))
            b = Fraction(match.group(3))
            area = a * b
            parsed_faces.append((face_num, area))
            total_area += area

        # Parse squares: "Face N is a square with side S cm."
        square_pattern = r"Face (\d+) is a square with side (\d+)\s*cm"
        for match in re.finditer(square_pattern, text):
            face_num = int(match.group(1))
            s = Fraction(match.group(2))
            area = s * s
            parsed_faces.append((face_num, area))
            total_area += area

        # Parse triangles: "Face N is a triangle with base B cm and height H cm."
        triangle_pattern = r"Face (\d+) is a triangle with base (\d+)\s*cm and height (\d+)\s*cm"
        for match in re.finditer(triangle_pattern, text):
            face_num = int(match.group(1))
            b = Fraction(match.group(2))
            h = Fraction(match.group(3))
            area = b * h / 2
            parsed_faces.append((face_num, area))
            total_area += area

        assert parsed_faces, f"No faces found in: {text}"

        # Verify we parsed all Face N mentions (the excluded face is NOT a "Face N is a..." sentence)
        face_count_in_faces = len(re.findall(r"Face \d+ is a (rectangle|square|triangle)", text))
        assert len(parsed_faces) == face_count_in_faces, (
            f"Face count mismatch: parsed {len(parsed_faces)}, found {face_count_in_faces} Face N sentences in {text}"
        )

        return total_area

    @pytest.mark.parametrize(
        "topic", ["find_the_surface_area_of_pyramids_such_as", "use_addition_of_the_area_of_the_faces"]
    )
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", range(60))
    def test_oracle(self, topic: str, difficulty: str, seed: int):
        """Test that independently derived answer matches the generated answer."""

        question = mathgen.make(topic, seed=seed, difficulty=difficulty)
        text = question.text

        # Derive answer based on topic
        if topic == "find_the_surface_area_of_pyramids_such_as":
            # Determine pyramid type from text content
            if "square base" in text:
                expected = self.parse_pyramid_square(text)
            elif "rectangular base" in text:
                expected = self.parse_pyramid_rectangular(text)
            elif "right-triangular base" in text:
                expected = self.parse_pyramid_right_triangle(text)
            else:
                pytest.fail(f"Cannot determine pyramid type for seed {seed}: {text}")

        elif topic == "use_addition_of_the_area_of_the_faces":
            expected = self.parse_faces(text)

        else:
            pytest.fail(f"Unknown topic: {topic}")

        # Assert answer is an integer
        assert expected.denominator == 1, (
            f"Expected integer answer for {topic} {difficulty} seed {seed}, got {expected}"
        )

        # Assert answer matches
        actual = Fraction(question.answer_text)
        assert expected == actual, (
            f"Mismatch for {topic} {difficulty} seed {seed}: expected {expected}, got {actual}\nQuestion: {text}"
        )

        # Assert all distractors are different from expected answer
        for distractor_text in question.distractors:
            distractor = Fraction(distractor_text)
            assert distractor != expected, (
                f"Distractor matches expected for {topic} {difficulty} seed {seed}: "
                f"distractor={distractor}, expected={expected}\n"
                f"Question: {text}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
