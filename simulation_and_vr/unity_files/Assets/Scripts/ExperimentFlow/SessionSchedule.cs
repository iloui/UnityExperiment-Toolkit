using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace Assets.Scripts.ExperimentFlow
{
    /// <summary>
    /// Pure scheduling logic of a participant session (no Unity calls): participant-ID validation, the balanced Latin
    /// square that counterbalances layout order, and the resulting trial list.
    /// </summary>
    public static class SessionSchedule
    {
        public struct Trial
        {
            /// <summary>0-based position in the session.</summary>
            public int trialIndex;
            /// <summary>Index into ParticipantSessionConfig.LayoutScenePaths.</summary>
            public int layoutIndex;
            /// <summary>Position of this layout in the participant's order (0 = walked first).</summary>
            public int orderPosition;
            /// <summary>0-based repetition of this layout for this participant; written as the manifest's repetition_index.</summary>
            public int repetitionIndex;
        }

        /// <summary>Null if the ID is usable, else the reason it is refused.</summary>
        public static string ValidateParticipantId(string id, string pattern)
        {
            if (string.IsNullOrWhiteSpace(id)) return "Enter a participant ID.";
            id = id.Trim();
            if (string.Equals(id, "unknown", StringComparison.OrdinalIgnoreCase)) return "'unknown' is not a participant ID.";
            if (!Regex.IsMatch(id, @"^[A-Za-z0-9_-]+$")) return "Use only letters, digits, '_' and '-'.";
            if (!string.IsNullOrEmpty(pattern) && !Regex.IsMatch(id, pattern)) return $"The ID must match {pattern} (e.g. P07).";
            if (!TryParticipantNumber(id, out _)) return "The ID must end in the participant number (e.g. P07).";
            return null;
        }

        /// <summary>The trailing number of the ID (P07 -> 7), which selects the Latin-square row.</summary>
        public static bool TryParticipantNumber(string id, out int number)
        {
            Match m = Regex.Match(id ?? "", @"(\d+)$");
            number = 0;
            return m.Success && int.TryParse(m.Groups[1].Value, out number);
        }

        /// <summary>
        /// Balanced Latin square (Williams design) for n conditions: every condition appears once at every position and
        /// follows every other condition equally often. n rows for even n, 2n rows (the n rows plus their reverses) for
        /// odd n - for n = 3 that is all 6 orders.
        /// </summary>
        public static int[][] BalancedLatinSquare(int n)
        {
            if (n < 1) throw new ArgumentOutOfRangeException(nameof(n));
            // First row 0, 1, n-1, 2, n-2, ...; row r adds r (mod n).
            var first = new int[n];
            for (int j = 0, lo = 1, hi = n - 1; j < n; j++)
            {
                if (j == 0) first[j] = 0;
                else if (j % 2 == 1) first[j] = lo++;
                else first[j] = hi--;
            }

            var rows = new List<int[]>();
            for (int r = 0; r < n; r++)
            {
                var row = new int[n];
                for (int j = 0; j < n; j++) row[j] = (first[j] + r) % n;
                rows.Add(row);
            }
            if (n % 2 == 1)
            {
                for (int r = 0; r < n; r++)
                {
                    var reversed = (int[])rows[r].Clone();
                    Array.Reverse(reversed);
                    rows.Add(reversed);
                }
            }
            return rows.ToArray();
        }

        /// <summary>Latin-square row for a participant number: P1 gets row 0, P2 row 1, ... cycling.</summary>
        public static int LatinSquareRow(int participantNumber, int rowCount)
        {
            return ((participantNumber - 1) % rowCount + rowCount) % rowCount;
        }

        /// <summary>
        /// The session's trials: layouts in the participant's Latin-square order, each walked 'repetitions' times in a
        /// row (blocked), so repetition_index 0..repetitions-1 counts the walks of that layout.
        /// </summary>
        public static List<Trial> BuildTrials(int participantNumber, int layoutCount, int repetitions, out int latinRow, out int[] layoutOrder)
        {
            int[][] square = BalancedLatinSquare(layoutCount);
            latinRow = LatinSquareRow(participantNumber, square.Length);
            layoutOrder = square[latinRow];

            var trials = new List<Trial>();
            for (int position = 0; position < layoutOrder.Length; position++)
            {
                for (int rep = 0; rep < repetitions; rep++)
                {
                    trials.Add(new Trial
                    {
                        trialIndex = trials.Count,
                        layoutIndex = layoutOrder[position],
                        orderPosition = position,
                        repetitionIndex = rep
                    });
                }
            }
            return trials;
        }
    }
}
