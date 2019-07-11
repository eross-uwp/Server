
# Server  
  
Contains datafiles and python scripts relating to different predictive models for student success.  
  

## Directories

 - **Data**
	 - Contains our main data files. Grades for all CSSE students who graduated or were dismissed between 2013-2018.
 - **bayesian_network**
	 - The WIP bayesian network.
 - **correlations_between_course_grades**
	 - Spearman's ranked correlations for all courses in our dataset. Also has Bonferroni adjusted p-values and graphs for seemingly significant course correlations.
 - **next_semester_gpa_prediction**
	 - Predicting a student's next term gpa based off of their current term gpa.
 - **nth_semester_graduation_prediction**
	 - Predicting whether a student will graduate or not based off of their 1st through nth term gpa. 
 - **postrequisite_prediction**
	 - Predicting a postrequisite course grade based on features related to grades from all immediate prerequisite courses.
