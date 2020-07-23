TreeScripts:
	Node: Class to represent a node in our course tree.
	TreeMaker: Class to generate a prerequisite structure for a given course name.
	FindGradeForCourse: Dataset generator used to get all grades for all retakes per student per course.
GenerateImmediate/All/RootPrereqTables: Used for our models. Creates a csv for each postrequisite that contains all
	immediate/all/root prerequisites and their grade for each. Also has features such as term/cumulative gpa and struggling status for
	the term previous to the oldest taken prerequisite.
PrereqToPostreqProbabilities:
	Calculating the likelihood of students passing/failing a prereq and taking/passing/failing its postreq.
Predict:
    For 'all' prediction, predict a postreq grade given grades from all prereqs (including prereqs of other prereqs)
    For 'immediate' prediction, predict a postreq grade given grades from all immediate prereqs
    For 'root' prediction, predict a postreq grade given grades from its lowest level prereqs. So 'all' prereqs that don't have prereqs themselves.
    Must have at least 25 students who took a postreq and at least one of its 'all'/'immediate'/'root' prereqs in order to run the model.
BayesNetPredict:
    Same as predict, but for the bayesian network. It was easier to make it its own thing than try to weasel it into the existing predict.
CombineResults:
    Combines the prediction results for each model into csvs to make adding them to the analysis files easy.
PValAnalysis:
    Calculates pvalue for each model pair. 
ExperimentRunner:
    Runs the experiments based on specific file formats. (Training.csv and Testing.csv are required, the rest are generated)
Ordinal Classifier:
    Neat little thing we never got around to trying. Based on this:
	https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c

***Note*** 
Results folder will be automatically generated, but if you run predictions it will not find the tuning folder unless you rename the one you want to use to 'TuningResults'
Also note, you will need to change a line of code in Predict.py to switch between with_gpa and only_prereqs. We are currently on only_prereqs.