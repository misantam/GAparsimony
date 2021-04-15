import numpy  as np
from src.parsimony_miscfun import clearConsoleLine


# new function for monitoring within RStudio
def parsimony_monitor(object, digits = 7, *args):
  fitnessval = object.fitnessval[~np.isnan(object.fitnessval)]
  fitnesstst = object.fitnesstst[~np.isnan(object.fitnesstst)]
  complexity = object.complexity[~np.isnan(object.complexity)]
  time_min = object.minutes_gen

  sumryStat = [np.mean(fitnessval), np.max(fitnessval), np.mean(fitnesstst), fitnesstst[np.argmax(fitnessval)], 
                            np.mean(complexity), complexity[np.argmax(fitnessval)], time_min]
  # sumryStat = round(sumryStat, digits)
  
  print(f"GA-PARSIMONY | iter ={object.iter}")
  print(f"MeanVal = {round(sumryStat[0], digits)}| ValBest = {object.bestfitnessVal}| TstBest = {object.bestfitnessTst}| ComplexBest = {object.bestcomplexity}| Time(min)= {object.minutes_gen}\n")
  # clearConsoleLine()



# Duda si es todo el rato con x1
# Equivalencia a fivenum es np.percentile(aux, [0, 25, 50, 75, 100])

def parsimony_summary(object, *args):
  
  x1 = object.fitnessval[~np.isnan(object.fitnessval)]
  q1 = np.percentile(x1, [0, 25, 50, 75, 100])
  x2 = object.fitnesstst[~np.isnan(object.fitnesstst)]
  q2 = np.percentile(x1, [0, 25, 50, 75, 100])
  x3 = object.complexity[~np.isnan(object.complexity)]
  q3 = np.percentile(x1, [0, 25, 50, 75, 100])
  # c(maxval = q1[4], meanval = mean(x1), q3val = q1[3], medianval = q1[2], q1val = q1[1], minval = q1[0],
  #   maxtst = q2[4], meantst = mean(x2), q3tst = q2[3], mediantst = q2[2], q1tst = q2[1], mintst = q2[0],
  #   maxcomplex = q3[4], meancomplex = mean(x3), q3complex = q3[3], mediancomplex = q3[2], q1complex = q3[1], mincomplex = q3[0])

  return q1[4], np.mean(x1), q1[3], q1[2], q1[1], q1[0], q2[4], np.mean(x2), q2[3], q2[2], q2[1], q2[0], q3[4], np.mean(x3), q3[3], q3[2], q3[1], q3[0]



