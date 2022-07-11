from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

heartdisease_model=BayesianModel([('age','heartdisease'),('cp','heartdisease'),('heartdisease','restecg'),('heartdisease','chol')])
print('Bayesian network models are :')
print('\t',heartdisease_model.nodes())
print('Bayesian edges are:')
print('\t',heartdisease_model.edges())

cpd_age = TabularCPD(variable='age', variable_card=2,
                      values=[[0.4], [0.6]])
#Age=0 (Young person below 30) Age =1 (Older person above 30) 
cpd_cp = TabularCPD(variable='cp', variable_card=2,
                       values=[[0.1], [0.9]])
#chestpain=0 (no chest pain), chestpain=1 (has chest pain)
cpd_heartdisease = TabularCPD(variable='heartdisease', variable_card=2,
                        values=[[0.99, 0.1, 0.8, 0.02],
                                [0.01, 0.9, 0.2, 0.98]],
                        evidence=['age', 'cp'],
                        evidence_card=[2, 2])
cpd_restecg = TabularCPD(variable='restecg', variable_card=2,
                      values=[[0.9, 0.2], [0.1, 0.8]],
                      evidence=['heartdisease'], evidence_card=[2])
#RestEcg=0 (Normal ECG) RestEcg=1 (Ubnormal ECG)
cpd_chol = TabularCPD(variable='chol', variable_card=2,
                      values=[[0.65, 0.3], [0.35, 0.7]],
                      evidence=['heartdisease'], evidence_card=[2])
#chol=0 (Normal Cholestrol) chol=1 (High Cholestrol)

# Associating the parameters with the model structure.
heartdisease_model.add_cpds(cpd_age, cpd_cp, cpd_heartdisease, cpd_restecg, cpd_chol)

# Checking if the cpds are valid for the model.
heartdisease_model.check_model()

heartdisease_infer=VariableElimination(heartdisease_model)

print('All local independecies are as follows')
heartdisease_model.get_independencies()
print('Displaying CPDs')
print(heartdisease_model.get_cpds('age'))
print(heartdisease_model.get_cpds('cp'))
print(heartdisease_model.get_cpds('heartdisease'))
print(heartdisease_model.get_cpds('restecg'))
print(heartdisease_model.get_cpds('chol'))

print('\n Probablity of heartdisease given chest pain')
q=heartdisease_infer.query(variables=['heartdisease'],evidence={'cp':1})
print(q)
