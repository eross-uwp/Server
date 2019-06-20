
def convert_term_number(semester, panda, index):
    if semester == 'fall':
        return convert_fall(panda, index)
    if semester == 'summer':
        return convert_summer(panda, index)
    if semester == 'winter':
        convert_winter(panda, index)
    if semester == 'spring':
        convert_spring(panda, index)


def convert_fall(panda, index):
    if panda.at[index, 'year'] == 2007:
        return 740
    if panda.at[index, 'year'] == 2008:
        return 770
    if panda.at[index, 'year'] == 2009:
        return 800
    if panda.at[index, 'year'] == 2010:
        return 830
    if panda.at[index, 'year'] == 2011:
        return 860
    if panda.at[index, 'year'] == 2012:
        return 890
    if panda.at[index, 'year'] == 2013:
        return 920
    if panda.at[index, 'year'] == 2014:
        return 950
    if panda.at[index, 'year'] == 2015:
        return 980
    if panda.at[index, 'year'] == 2016:
        return 1010
    if panda.at[index, 'year'] == 2017:
        return 1040
    if panda.at[index, 'year'] == 2018:
        return 1070


def convert_summer(panda, index):
    if panda.at[index, 'year'] == 2008:
        return 760
    if panda.at[index, 'year'] == 2009:
        return 790
    if panda.at[index, 'year'] == 2010:
        return 820
    if panda.at[index, 'year'] == 2011:
        return 850
    if panda.at[index, 'year'] == 2012:
        return 880
    if panda.at[index, 'year'] == 2013:
        return 910
    if panda.at[index, 'year'] == 2014:
        return 940
    if panda.at[index, 'year'] == 2015:
        return 970
    if panda.at[index, 'year'] == 2016:
        return 1000
    if panda.at[index, 'year'] == 2017:
        return 1030
    if panda.at[index, 'year'] == 2018:
        return 1060


def convert_winter(panda, index):
    if panda.at[index, 'year'] == 2008:
        return 745
    if panda.at[index, 'year'] == 2009:
        return 775
    if panda.at[index, 'year'] == 2010:
        return 805
    if panda.at[index, 'year'] == 2011:
        return 835
    if panda.at[index, 'year'] == 2012:
        return 865
    if panda.at[index, 'year'] == 2013:
        return 895
    if panda.at[index, 'year'] == 2014:
        return 925
    if panda.at[index, 'year'] == 2015:
        return 955
    if panda.at[index, 'year'] == 2016:
        return 985
    if panda.at[index, 'year'] == 2017:
        return 1015
    if panda.at[index, 'year'] == 2018:
        return 1045
    if panda.at[index, 'year'] == 2019:
        return 1075


def convert_spring(panda, index):
    if panda.at[index, 'year'] == 2008:
        return 750
    if panda.at[index, 'year'] == 2009:
        return 780
    if panda.at[index, 'year'] == 2010:
        return 810
    if panda.at[index, 'year'] == 2011:
        return 840
    if panda.at[index, 'year'] == 2012:
        return 870
    if panda.at[index, 'year'] == 2013:
        return 900
    if panda.at[index, 'year'] == 2014:
        return 930
    if panda.at[index, 'year'] == 2015:
        return 960
    if panda.at[index, 'year'] == 2016:
        return 990
    if panda.at[index, 'year'] == 2017:
        return 1020
    if panda.at[index, 'year'] == 2018:
        return 1050
    if panda.at[index, 'year'] == 2019:
        return 1080
