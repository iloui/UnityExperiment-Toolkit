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
            /// <summary>Position of the task within the layout: 0..trainingTasks-1 = training tasks, trainingTasks = the test task (last).</summary>
            public int taskOrderPosition;
            /// <summary>Index into HumanTaskSet.training, or -1 for the test task.</summary>
            public int trainingTaskIndex;
            /// <summary>Latin-square row that ordered this layout's training tasks.</summary>
            public int taskOrderLatinRow;

            public bool IsTest => trainingTaskIndex < 0;
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

        /// <summary>Participant fold (methodology 3.8.2): odd participant numbers are fold 1, even ones fold 2.</summary>
        public static int FoldOf(int participantNumber)
        {
            return participantNumber % 2 != 0 ? 1 : 2;
        }

        /// <summary>
        /// The session's trials (methodology 3.8.2): layouts in the participant's Latin-square order; within each layout the
        /// training tasks in the order of Latin-square row (participant number + layout index), then the test task last. The
        /// layout index offset keeps the task order from being tied to the layout order.
        /// </summary>
        public static List<Trial> BuildTrials(int participantNumber, int layoutCount, int trainingTasks, out int latinRow, out int[] layoutOrder)
        {
            int[][] layoutSquare = BalancedLatinSquare(layoutCount);
            latinRow = LatinSquareRow(participantNumber, layoutSquare.Length);
            layoutOrder = layoutSquare[latinRow];
            int[][] taskSquare = BalancedLatinSquare(trainingTasks);

            var trials = new List<Trial>();
            for (int position = 0; position < layoutOrder.Length; position++)
            {
                int layout = layoutOrder[position];
                int taskRow = LatinSquareRow(participantNumber + layout, taskSquare.Length);
                int[] taskOrder = taskSquare[taskRow];
                for (int k = 0; k <= trainingTasks; k++)
                {
                    trials.Add(new Trial
                    {
                        trialIndex = trials.Count,
                        layoutIndex = layout,
                        orderPosition = position,
                        taskOrderPosition = k,
                        trainingTaskIndex = k < trainingTasks ? taskOrder[k] : -1,
                        taskOrderLatinRow = taskRow
                    });
                }
            }
            return trials;
        }
    }
}
