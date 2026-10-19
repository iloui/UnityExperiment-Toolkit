namespace Assets.Scripts.Editor
{
    /// <summary>
    /// Paths and priorities of every toolkit entry under Unity's Tools menu, in one place so the menu reads as the workflow:
    ///   Generative Design          - generate layout scenes
    ///   Scene Preparation          - make a layout scene ready for either agent
    ///   Shortest Path Agent        - run and visualize the NavMesh baseline (the original toolkit's agent-based model)
    ///   Imitation Learning Agent   - steps 1-5 of the imitation pipeline (methodology 3.1, after scene preparation), then Advanced
    ///   Scene Utilities            - generic editing helpers
    /// Unity sorts menu items by priority and draws a separator where two neighbours differ by more than 10, so the
    /// priorities inside a section stay within 10 of each other except where a separator is wanted.
    /// </summary>
    internal static class ToolsMenu
    {
        private const string Root = "Tools/";

        // Generative Design
        public const string GenerativeDesign = Root + "Generative Design/";
        public const int GenerativeDesignPriority = 100;

        // Scene Preparation: shared by both agents.
        public const string ScenePreparation = Root + "Scene Preparation/";
        public const string PrepareScene = ScenePreparation + "Prepare Scene";
        public const int ScenePreparationPriority = 200;

        // Shortest Path Agent
        public const string ShortestPath = Root + "Shortest Path Agent/";
        public const int ShortestPathPriority = 300;

        // Imitation Learning Agent: one submenu per pipeline step.
        public const string Imitation = Root + "Imitation Learning Agent/";
        public const string SyntheticWalks = Imitation + "1 · Generate Synthetic Walks/";
        public const string HumanWalks = Imitation + "2 · Record Human Walks/";
        public const string TrainModel = Imitation + "3 · Train Model/";
        public const string RunAgent = Imitation + "4 · Run Imitation Agent/";
        public const string EvaluateDesign = Imitation + "5 · Evaluate Design/";
        public const string ImitationAdvanced = Imitation + "Advanced/";
        public const int SyntheticWalksPriority = 400;
        public const int HumanWalksPriority = 403;
        public const int TrainModelPriority = 408;
        public const int RunAgentPriority = 411;
        public const int EvaluateDesignPriority = 413;
        public const int ImitationAdvancedPriority = 430;

        // Scene Utilities
        public const string Utilities = Root + "Scene Utilities/";
        public const string Select = Utilities + "Select/";
        public const string Objects = Utilities + "Objects/";
        public const int SelectPriority = 1000;
        public const int ObjectsPriority = 1200;
        public const int UtilitiesWindowPriority = 1300;
    }
}
