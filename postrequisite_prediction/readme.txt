TreeScripts:
	Node: Class to represent a node in our course tree.
	TreeMaker: Class to generate a prerequisite structure for a given course name.
	FindGradeForCourse: Dataset generator used to get all grades for all retakes per student per course.
GenerateImmediatePrereqTables: Used for our models. Creates a csv for each postrequisite that contains all
	immediate prerequisites and their grade for each. Also has features such as term/cumulative gpa and struggling status for 
	the term previous to the oldest taken prerequisite.
PrereqToPostreqProbabilities:
	Calculating the likelihood of students passing/failing a prereq and taking/passing/failing its postreq.
All/Immediate/Root Predictions:
    For 'all' prediction, predict a postreq grade given grades from all prereqs (including prereqs of other prereqs)
    For 'immediate' prediction, predict a postreq grade given grades from all immediate prereqs
    For 'root' prediction, predict a postreq grade given grades from its lowest level prereqs. So 'all' prereqs that don't have prereqs themselves.
    Must have at least 25 students who took a postreq and at least one of its 'all'/'immediate'/'root' prereqs in order to run the model.