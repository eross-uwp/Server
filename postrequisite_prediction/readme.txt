TreeScripts:
	Node: Class to represent a node in our course tree.
	TreeMaker: Class to generate a prerequisite structure for a given course name.
	FindGradeForCourse: Dataset generator used to get all grades for all retakes per student per course.
GenerateImmediatePrereqTables: Used for our Logistic Regression model. Creates a csv for each postrequisite that contains all
	immediate prerequisites and their grade for each. Also has features such as term/cumulative gpa and struggling status for 
	the term previous to the oldest taken prerequisite. 
ImmediatePRereqsLogisticRegression: Uses the tables for each postreq to run a a logistic regression for each postreq. 5 fold stratification is
	done before the model is run. Predictions, and probabilities for each grade, are stored in a csv for each postreq.
PrereqToPostreqProbabilities:
	Calculating the likelihood of students passing/failing a prereq and taking/passing/failing its postreq.