from evaluate_class import MApMetric

metric = MApMetric(ovp_thresh, use_difficult, class_names)
results = mod.score(eval_iter, metric, num_batch=None)
for k, v in results:
    print("{}: {}".format(k, v))
