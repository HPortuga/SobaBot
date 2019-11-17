# Logs the algorithm's RandomSearchCV result
def writeParameterTuningLog(scores, algorithm, params):
  log = open("./Logs/"+algorithm+".txt", "w")

  log.write("Parameter Tuning: %s" % algorithm)
  log.write("\n============================\n")
  log.write("The following parameters were tuned to the algorithm with exhaustive search:\n")
  
  for param in params:
    log.write("%s: %s\n" % (param, params[param]))

  log.write("\n============================\n")
  log.write("The results are:\n\n")

  i = 0
  for score in scores:
    log.write("%d\n" % i)
    log.write("  Mean: %f, Std: %f\n  Params: %s\n\n" % (score["mean"], score["std"], str(score["params"])))
    i += 1

  log.close()
