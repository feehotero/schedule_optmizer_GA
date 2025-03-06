
def print_population(pop, subjects):
  for k in range (10):
    print_days()
    for j in range(6):
      column_values = []
      for i in range(5):
        column_values.append(subjects[pop[k][i][j]])
      print(" ".join(f"{value:20}" for value in column_values))
    print()

def print_solution(pop, subjects, fitness):
  print_days()
  for j in range(6):
    column_values = []
    for i in range(5):
      column_values.append(subjects[pop[fitness[0][0].astype(int)][i][j]])
    print(" ".join(f"{value:20}" for value in column_values))

def print_days():
  days = ['SEGUNDA', 'TERCA', 'QUARTA', 'QUINTA', 'SEXTA']
  print(" ".join(f'{day:20}' for day in days))
  print('-'*100)

def print_scores(score):
  print(f'Best: {score[0][1]} - (%): {score[0][2]}')
  print(f'Average: {score[len(score)//2][1]} - (%): {score[len(score)//2][2]}')
  print(f'Worst: {score[-1][1]} - (%): {score[-1][2]}')
  print()
