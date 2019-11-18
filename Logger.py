# Logs the algorithm's RandomSearchCV result
def writeParameterTuningLog(scores, algorithm, params):
  log = open("./Logs/Calibration/"+algorithm+".txt", "w")

  log.write("Parameter Tuning: %s" % algorithm)
  log.write("\n============================\n")
  log.write("The following parameters were tuned to the algorithm with random search:\n")
  
  for param in params:
    log.write("%s: %s\n" % (param, params[param]))

  log.write("============================\n")
  log.write("The results are:\n\n")

  i = 0
  for score in scores:
    log.write("%d\n" % i)
    log.write("  Acuracy: %.2f, Std: %.2f\n  Params: %s\n\n" % (score["mean"], score["std"], str(score["params"])))
    i += 1

  log.close()

# Logs the sampling with Leave One Out
def writeLooScores(scores, algorithm):
  log = open("./Logs/LeaveOneOut/"+algorithm+".txt", "w")

  log.write("Showing scores for sampling with Leave One Out running %s" % algorithm)
  log.write("The results are:\n\n")

  i = 0
  for score in scores:
    log.write("%d\n" % i)
    log.write("  Accuracy: %.2f, Precision: %.2f, Recall: %.2f\n  Params: %s\n\n" % (score["accuracy"], score["precision"], score["recall"], str(score["params"])))
    i += 1

  log.close()

def writeFinalPerformance(scores, algorithm, params):
  log = open("./Logs/Performance/"+algorithm+".txt", "w")

  log.write("Showing overall performance score for %s" % algorithm)
  log.write("\n============================\n")
  log.write("The following parameter got the best result while tuning:\n")
  
  log.write("%s\n" % (params))

  log.write("============================\n")
  log.write("The results are:\n\n")

  log.write("  Accuracy: %.2f, Precision: %.2f, Recall: %.2f\n\n" % (scores[0]["accuracy"], scores[0]["precision"], scores[0]["recall"]))

  log.close()
  
  
