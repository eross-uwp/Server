"""
__Author__: Nate Braukhoff
"""


def convert_term_number(semester, year):
    """
    Each term of the school year has a unique number attached to it. This method will assign tha number to the right
    term
    :param semester:
    :param year:
    :return: The term number
    """
    if semester == 'fall':
        return convert_fall(year)
    if semester == 'summer':
        return convert_summer(year)
    if semester == 'winter':
        return convert_winter(year)
    if semester == 'spring':
        return convert_spring(year)


def convert_fall(year):
    """
    All term number for fall semesters for years 07 - 18
    param year:
    :return: Term number for the fall semester
    """

    if year == 2007:
        return 740
    if year == 2008:
        return 770
    if year == 2009:
        return 800
    if year == 2010:
        return 830
    if year == 2011:
        return 860
    if year == 2012:
        return 890
    if year == 2013:
        return 920
    if year == 2014:
        return 950
    if year == 2015:
        return 980
    if year == 2016:
        return 1010
    if year == 2017:
        return 1040
    if year == 2018:
        return 1070


def convert_summer(year):
    """
    All term number for summer classes for years 07 - 18
    param year:
    :return: Term number for the fall semester
    """

    if year == 2008:
        return 760
    if year == 2009:
        return 790
    if year == 2010:
        return 820
    if year == 2011:
        return 850
    if year == 2012:
        return 880
    if year == 2013:
        return 910
    if year == 2014:
        return 940
    if year == 2015:
        return 970
    if year == 2016:
        return 1000
    if year == 2017:
        return 1030
    if year == 2018:
        return 1060


def convert_winter(year):
    """
    All term number for winter classes for years 07 - 18
    param year:
    :return: Term number for the fall semester
    """

    if year == 2008:
        return 745
    if year == 2009:
        return 775
    if year == 2010:
        return 805
    if year == 2011:
        return 835
    if year == 2012:
        return 865
    if year == 2013:
        return 895
    if year == 2014:
        return 925
    if year == 2015:
        return 955
    if year == 2016:
        return 985
    if year == 2017:
        return 1015
    if year == 2018:
        return 1045
    if year == 2019:
        return 1075


def convert_spring(year):
    """
    All term number for spring semesters for years 08 - 19
    param year:
    :return: Term number for the fall semester
    """
    if year == 2008:
        return 750
    if year == 2009:
        return 780
    if year == 2010:
        return 810
    if year == 2011:
        return 840
    if year == 2012:
        return 870
    if year == 2013:
        return 900
    if year == 2014:
        return 930
    if year == 2015:
        return 960
    if year == 2016:
        return 990
    if year == 2017:
        return 1020
    if year == 2018:
        return 1050
    if year == 2019:
        return 1080
