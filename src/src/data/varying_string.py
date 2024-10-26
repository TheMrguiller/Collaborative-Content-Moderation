# -*- coding: utf-8 -*-


class VaryingString:
    """Represents a string with varying character representations."""

    def __init__(self, string, char_map={}):
        """
        Args:
            string (str): String to generate variants of.
            char_map (dict): Maps characters to substitute characters.
        """
        self._original = string
        self._char_map = char_map
        self._char_combos = []
        self._min_len = 0
        self._max_len = 0
        self._original_word = string  # Store the original word

        # Create list of all possible character combinations.
        for char in self._original:
            if char in char_map:
                self._char_combos.append(char_map[char])
                lens = [len(c) for c in char_map[char]]
                self._min_len += min(lens)
                self._max_len += max(lens)
            else:
                self._char_combos.append((char,))
                self._min_len += 1
                self._max_len += 1

    def __str__(self):
        return self._original

    def __eq__(self, other):
        if self is other:
            return True
        elif isinstance(other, VaryingString):
            raise NotImplementedError
        elif isinstance(other, str):
            len_other = len(other)
            if len_other < self._min_len or len_other > self._max_len:
                return False
            slices = [other]
            for chars in self._char_combos:
                new_slices = []
                for char in chars:
                    if not char:
                        new_slices.extend(slices)
                    len_char = len(char)
                    for sl in slices:
                        if sl[:len_char] == char:
                            new_slices.append(sl[len_char:])
                if len(new_slices) == 0:
                    return False
                slices = new_slices
            for sl in slices:
                if len(sl) == 0:
                    return True
            return False
        else:
            return False

    def generate_variations(self):
        """Generate all possible variations of the string."""
        return self._generate_combinations(self._char_combos)

    def _generate_combinations(self, char_combos, current_combo=''):
        """Recursively generate all combinations of characters."""
        if not char_combos:
            return [current_combo]
        
        first_char_options = char_combos[0]
        remaining_combos = char_combos[1:]

        combinations = []
        for char in first_char_options:
            combinations.extend(self._generate_combinations(remaining_combos, current_combo + char))
        
        return combinations
