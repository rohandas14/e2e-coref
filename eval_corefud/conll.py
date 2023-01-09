import re
import os
import tempfile
import subprocess
import operator
import collections
import time

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")
COREF_RESULTS_REGEX = re.compile(r"(?<=CoNLL score: )\d*.?\d*", re.DOTALL)

def output_corefud(input_file, output_file, predictions, subtoken_map):
  prediction_map = {}
  for doc_key, clusters in predictions.items():
    start_map = collections.defaultdict(list)
    end_map = collections.defaultdict(list)
    word_map = collections.defaultdict(list)
    for cluster_id, mentions in enumerate(clusters):
      for start, end in mentions:
        # map sub-tokens to words
        start, end = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
        if start == end:
          word_map[start].append(cluster_id)
        else:
          start_map[start].append((cluster_id, end))
          end_map[end].append((cluster_id, start))
    for k,v in start_map.items():
      start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
    for k,v in end_map.items():
      end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
    prediction_map[doc_key] = (start_map, end_map, word_map)

  word_index = 0
  for line in input_file.readlines():
    row = line.split("\t")
    if len(row) == 0:
      output_file.write("\n")
    elif len(row) == 1 and row[0] == "\n":
      output_file.write("\n")
    elif row[0].startswith("#"):
      if row[0].startswith("# newdoc"):
        doc_key = row[0].split()[-1]
        start_map, end_map, word_map = prediction_map[doc_key]
        word_index = 0
      output_file.write(line)
    else:
      coref_list = []
      if word_index in end_map:
        for cluster_id in end_map[word_index]:
          coref_list.append("e{})".format(cluster_id))
      if word_index in word_map:
        for cluster_id in word_map[word_index]:
          coref_list.append("(e{}--1-)".format(cluster_id))
      if word_index in start_map:
        for cluster_id in start_map[word_index]:
          coref_list.append("(e{}--1-".format(cluster_id))

      if len(coref_list) == 0:
        row[-1] = "_"
      else:
        row[-1] = "Entity=" + "".join(coref_list)

      output_file.write("\t".join(row))
      output_file.write("\n")
      word_index += 1

def official_corefud_eval(gold_path, predicted_path):
  cmd = ["python3", "./corefud-scorer/corefud-scorer.py", gold_path, predicted_path]
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
  stdout, stderr = process.communicate()
  process.wait()

  stdout = stdout.decode("utf-8")
  if stderr is not None:
    for line in stderr.readlines():
      if line.startswith("WARNING"):
        continue
      else:
        print(line)

  coref_results_match = re.search(COREF_RESULTS_REGEX, stdout)
  return float(coref_results_match.group(0).strip())

def evaluate_conll(gold_corefud_path, predictions_path, predictions, subtoken_map):
  # with tempfile.NamedTemporaryFile(delete=True, mode='w') as prediction_file:
  #   with open(gold_corefud_path, "r") as gold_file:
  #     output_corefud(gold_file, prediction_file, predictions, subtoken_map)
  # return official_corefud_eval(gold_file.name, prediction_file.name)
  
  with open(predictions_path, "w") as prediction_corefud_file:
    with open(gold_corefud_path, "r") as gold_corefud_file:
      output_corefud(gold_corefud_file, prediction_corefud_file, predictions, subtoken_map)
    print("file: " + prediction_corefud_file.name, flush=True)
  return official_corefud_eval(gold_corefud_file.name, prediction_corefud_file.name)
