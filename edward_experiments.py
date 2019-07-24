from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from edward.models import Gamma, Poisson, Normal, PointMass, \
    TransformedDistribution
import pandas as pd
#from observations import insteval

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pdb


def build_lin_reg_toy_dataset(N, w, noise_std):
  D = len(w)
  x = np.random.randn(N, D)
  y = np.dot(x, w) + np.random.normal(0, noise_std, size=N)
  return x, y

def bayesian_linear_regression():
  # underlying model params
  N = 5000  # number of data points
  D = 100  # number of features
  noise_std = .1
  # Generate simulated data
  w_true = np.random.randn(D)
  X_train, y_train = build_lin_reg_toy_dataset(N, w_true, noise_std)
  X_test, y_test = build_lin_reg_toy_dataset(N, w_true, noise_std)
  # Set up edward model
  X = tf.placeholder(tf.float32, [N, D])
  w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
  b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
  log_sd = Normal(loc=tf.zeros(1), scale=tf.ones(1))
  y = Normal(loc=ed.dot(X, w) + b, scale=tf.exp(log_sd))
  # Inference in edward
  qw = Normal(loc=tf.get_variable("qw/loc", [D]),
            scale=tf.nn.softplus(tf.get_variable("qw/scale", [D])))
  qb = Normal(loc=tf.get_variable("qb/loc", [1]),
            scale=tf.nn.softplus(tf.get_variable("qb/scale", [1])))
  qlog_sd = Normal(loc=tf.get_variable("qlog_sd/loc", [1]),
            scale=tf.nn.softplus(tf.get_variable("qlog_sd/scale", [1])))
  inference = ed.KLqp({w: qw, b: qb, log_sd: qlog_sd}, data={X: X_train, y: y_train})

  inference.run(n_iter=1000)
  pdb.set_trace()
  #qw.mean().eval()


def build_matrix_factorization_toy_dataset(U, V, N, M, noise_std):
  noises = []
  R = np.dot(U, V) #+ np.random.normal(0, noise_std, size=(N, M))
  gene_residual_sdevs = np.sqrt(np.random.exponential(size=M))
  for m in range(M):
    R[:,m] = R[:, m] + np.random.normal(0, gene_residual_sdevs[m],size=N)
  return R, noises


def bayesian_matrix_factorization():
  N = 10000
  M = 5000
  D = 3
  noise_std = .1

  # true latent factors
  U_true = np.random.randn(N, D)
  V_true = np.random.randn(D, M)
  # DATA
  R_true, noises = build_matrix_factorization_toy_dataset(U_true, V_true, N, M, noise_std)

  print('data laoded')

  # MODEL
  U = Normal(loc=0.0, scale=1.0, sample_shape=[N, D])
  V = Normal(loc=0.0, scale=1.0, sample_shape=[D, M])
  log_sd = Normal(loc=tf.zeros(M), scale=tf.ones(M))
  R = Normal(loc=tf.matmul(U, V),
             scale=tf.matmul(tf.ones([N,M]),tf.matrix_diag(tf.exp(log_sd))))

  # INFERENCE
  qU = Normal(loc=tf.get_variable("qU/loc", [N, D]),
              scale=tf.nn.softplus(
                  tf.get_variable("qU/scale", [N, D])))
  qV = Normal(loc=tf.get_variable("qV/loc", [D, M]),
              scale=tf.nn.softplus(
                  tf.get_variable("qV/scale", [D, M])))
  qlog_sd = Normal(loc=tf.get_variable("qlog_sd/loc", [M]),
            scale=tf.nn.softplus(tf.get_variable("qlog_sd/scale", [M])))
  inference = ed.KLqp({U: qU, V: qV, log_sd: qlog_sd}, data={R: R_true})
  inference.run()
  pdb.set_trace()


def bayesian_linear_mixed_model():
  ###########
  # Load in data
  ############
  data, metadata = insteval("~/data")
  data = pd.DataFrame(data, columns=metadata['columns'])
  # s - students - 1:2972
  # d - instructors - codes that need to be remapped
  # dept also needs to be remapped
  data['s'] = data['s'] - 1
  data['dcodes'] = data['d'].astype('category').cat.codes
  data['deptcodes'] = data['dept'].astype('category').cat.codes
  data['y'] = data['y'].values.astype(float)

  train = data.sample(frac=0.8)
  test = data.drop(train.index)
  s_train = train['s'].values
  
  d_train = train['dcodes'].values
  dept_train = train['deptcodes'].values
  y_train = train['y'].values
  service_train = train['service'].values
  n_obs_train = train.shape[0]

  s_test = test['s'].values
  d_test = test['dcodes'].values
  dept_test = test['deptcodes'].values
  y_test = test['y'].values
  service_test = test['service'].values
  n_obs_test = test.shape[0]


  n_s = max(s_train) + 1  # number of students
  n_d = max(d_train) + 1  # number of instructors
  n_dept = max(dept_train) + 1  # number of departments
  n_obs = train.shape[0]  # number of observations

  ###############################
  ## MODEL
  # y ~ 1 + (1|students) + (1|instructor) + (1|dept) + service
  ###############################
  # Set up placeholders for the data inputs.
  s_ph = tf.placeholder(tf.int32, [None])
  d_ph = tf.placeholder(tf.int32, [None])
  dept_ph = tf.placeholder(tf.int32, [None])
  service_ph = tf.placeholder(tf.float32, [None])
  # Set up fixed effects.
  mu = tf.get_variable("mu", [])
  service = tf.get_variable("service", [])

  sigma_s = tf.sqrt(tf.exp(tf.get_variable("sigma_s", [])))
  sigma_d = tf.sqrt(tf.exp(tf.get_variable("sigma_d", [])))
  sigma_dept = tf.sqrt(tf.exp(tf.get_variable("sigma_dept", [])))

  # Set up random effects.
  eta_s = Normal(loc=tf.zeros(n_s), scale=sigma_s * tf.ones(n_s))
  eta_d = Normal(loc=tf.zeros(n_d), scale=sigma_d * tf.ones(n_d))
  eta_dept = Normal(loc=tf.zeros(n_dept), scale=sigma_dept * tf.ones(n_dept))

  yhat = (tf.gather(eta_s, s_ph) +
        tf.gather(eta_d, d_ph) +
        tf.gather(eta_dept, dept_ph) +
        mu + service * service_ph)
  y = Normal(loc=yhat, scale=tf.ones(n_obs))


  ###############################
  ## Inference set up
  ###############################
  q_eta_s = Normal(
    loc=tf.get_variable("q_eta_s/loc", [n_s]),
    scale=tf.nn.softplus(tf.get_variable("q_eta_s/scale", [n_s])))
  q_eta_d = Normal(
    loc=tf.get_variable("q_eta_d/loc", [n_d]),
    scale=tf.nn.softplus(tf.get_variable("q_eta_d/scale", [n_d])))
  q_eta_dept = Normal(
    loc=tf.get_variable("q_eta_dept/loc", [n_dept]),
    scale=tf.nn.softplus(tf.get_variable("q_eta_dept/scale", [n_dept])))

  latent_vars = {
    eta_s: q_eta_s,
    eta_d: q_eta_d,
    eta_dept: q_eta_dept}
  data = {
    y: y_train,
    s_ph: s_train,
    d_ph: d_train,
    dept_ph: dept_train,
    service_ph: service_train}
  inference = ed.KLqp(latent_vars, data)
  tf.global_variables_initializer().run()
  inference.run()

  #for _ in range(inference.n_iter):
    # Update and print progress of algorithm.
    #info_dict = inference.update()
    #inference.print_progress(info_dict)

    #t = info_dict['t']

  pdb.set_trace()


def generate_multivariate_bayesian_linear_mixed_model_data(num_genes, num_individuals, cells_per_individual):
  # Number of samples in our model is the product of the number of individuals and the number of cells for each individual
  num_samples = num_individuals*cells_per_individual
  # Initialize simulated gene expression matrix
  Y = np.zeros((num_samples, num_genes))
  # Initialize simulated matrix containing random effects for each individual
  random_effects = np.zeros((num_individuals, num_genes))
  Z = []
  # Fill in matrix keeping track of which individual each sample belongs to
  sample_num = 0
  for indi_num in range(num_individuals):
    for cell_num in range(cells_per_individual):
      Z.append([indi_num])
      sample_num = sample_num + 1
  # Simulate random effect stdevs for each gene
  gene_random_effects_sdevs =  np.sqrt(np.random.exponential(size=num_genes))
  # Simulate resdiaul effect sdevs for each gene
  gene_residual_sdevs = np.sqrt(np.random.exponential(size=num_genes))
  # Simulate mean for each gene
  gene_mean = np.random.normal(size=num_genes)
  # Simulate gene expression data for each gene
  for gene_num in range(num_genes):
    # Simulate random effects for each individual
    gene_random_effect = np.random.normal(loc=0, scale=gene_random_effects_sdevs[gene_num], size=num_individuals)
    random_effects[:, gene_num] = gene_random_effect
    # Loop through samples
    for sample_num in range(num_samples):
      # Get index of individual corresponding to this sample
      indi_id = Z[sample_num][0]
      # Get simulated predicted mean for this sample
      predicted_mean = gene_mean[gene_num] + gene_random_effect[int(indi_id)]
      # Randomly draw expression sample for this sample
      Y[sample_num, gene_num] = np.random.normal(loc=predicted_mean, scale=gene_residual_sdevs[gene_num])
  return Y, Z, random_effects, gene_random_effects_sdevs, gene_residual_sdevs



def multivariate_bayesian_linear_mixed_model():
  ###########
  # Load/Simulate in data
  ############
  num_genes = 5000
  num_individuals = 100
  cells_per_individual = 100
  num_samples = num_individuals*cells_per_individual
  print('start loading')
  Y_train, Z_train, true_random_effects, true_gene_random_effects_sdevs, true_gene_residual_sdevs = generate_multivariate_bayesian_linear_mixed_model_data(num_genes, num_individuals, cells_per_individual)
  print('data loaded')
  ###############################
  ## MODEL
  # Y ~ 1 + (1|individual) 
  ###############################
  # Set up placeholders for the data inputs.
  ind_ph = tf.placeholder(tf.int32, [num_samples, 1])
  # Set up fixed effects.
  mu = tf.get_variable("mu", [num_genes])

  sigma_ind = tf.sqrt(tf.exp(tf.get_variable("sigma_ind", [num_genes])))

  sigma_resid = tf.sqrt(tf.exp(tf.get_variable("sigma_resid", [num_genes])))


  # Set up random effects
  eta_ind = Normal(loc=tf.zeros([num_individuals, num_genes]), scale= tf.matmul(tf.ones([num_individuals,num_genes]),tf.matrix_diag(sigma_ind)))


  yhat = (tf.gather_nd(eta_ind, ind_ph) + tf.matmul(tf.ones([num_samples, num_genes]), tf.matrix_diag(mu)))
  y = Normal(loc=yhat, scale=tf.matmul(tf.ones([num_samples, num_genes]), tf.matrix_diag(sigma_resid)))

  ###############################
  ## Inference set up
  ###############################
  q_ind_s = Normal(
    loc=tf.get_variable("q_ind_s/loc", [num_individuals, num_genes]),
    scale=tf.nn.softplus(tf.get_variable("q_ind_s/scale", [num_individuals, num_genes])))

  latent_vars = {
    eta_ind: q_ind_s}
  data = {
    y: Y_train,
    ind_ph: Z_train}
  inference = ed.KLqp(latent_vars, data)
  tf.global_variables_initializer().run()
  inference.run()
  pdb.set_trace()




def generate_multivariate_bayesian_linear_mixed_model_and_factorization_data(num_genes, num_individuals, cells_per_individual, K):
  # Number of samples in our model is the product of the number of individuals and the number of cells for each individual
  num_samples = num_individuals*cells_per_individual
  # Initialize simulated gene expression matrix
  Y = np.zeros((num_samples, num_genes))
  # Initialize simulated matrix containing random effects for each individual
  random_effects = np.zeros((num_individuals, num_genes))
  Z = []
  # Fill in matrix keeping track of which individual each sample belongs to
  sample_num = 0
  for indi_num in range(num_individuals):
    for cell_num in range(cells_per_individual):
      Z.append([indi_num])
      sample_num = sample_num + 1
  # Simulate matrix factorization
  U_true = np.abs(np.random.randn(num_samples, K))
  V_true = np.random.randn(K, num_genes)
  R_true = np.dot(U_true, V_true)
  Y = Y + R_true
  # Simulate random effect stdevs for each gene
  gene_random_effects_sdevs =  np.sqrt(np.random.exponential(size=num_genes))
  # Simulate resdiaul effect sdevs for each gene
  gene_residual_sdevs = np.sqrt(np.random.exponential(size=num_genes))
  # Simulate mean for each gene
  gene_mean = np.random.normal(size=num_genes)
  # Simulate gene expression data for each gene
  for gene_num in range(num_genes):
    # Simulate random effects for each individual
    gene_random_effect = np.random.normal(loc=0, scale=gene_random_effects_sdevs[gene_num], size=num_individuals)
    random_effects[:, gene_num] = gene_random_effect
    # Loop through samples
    for sample_num in range(num_samples):
      # Get index of individual corresponding to this sample
      indi_id = Z[sample_num][0]
      # Get simulated predicted mean for this sample
      predicted_mean = gene_mean[gene_num] + gene_random_effect[int(indi_id)]
      # Randomly draw expression sample for this sample
      Y[sample_num, gene_num] = Y[sample_num, gene_num] + np.random.normal(loc=predicted_mean, scale=gene_residual_sdevs[gene_num])
  return Y, Z, random_effects, gene_random_effects_sdevs, gene_residual_sdevs, U_true, V_true, gene_mean


def lognormal_q(shape, name=None):
  with tf.variable_scope(name, default_name="lognormal_q"):
    min_scale = 1e-5
    loc = tf.get_variable("loc", shape)
    scale = tf.get_variable(
        "scale", shape, initializer=tf.random_normal_initializer(stddev=0.1))
    rv = TransformedDistribution(
        distribution=Normal(loc, tf.maximum(tf.nn.softplus(scale), min_scale)),
        bijector=tf.contrib.distributions.bijectors.Exp())
    return rv


def multivariate_bayesian_linear_mixed_model_and_factorization():
  ###########
  # Load/Simulate in data
  ############
  num_genes = 5000
  num_individuals = 100
  cells_per_individual = 100
  K = 3
  num_samples = num_individuals*cells_per_individual
  print('start loading')
  Y_train, Z_train, true_random_effects, true_gene_random_effects_sdevs, true_gene_residual_sdevs, U_true, V_true, gene_mean = generate_multivariate_bayesian_linear_mixed_model_and_factorization_data(num_genes, num_individuals, cells_per_individual, K)
  print('data loaded')
  ###############################
  ## MODEL
  # Y ~ 1 + (1|individual) 
  ###############################
  # Set up placeholders for the data inputs.
  ind_ph = tf.placeholder(tf.int32, [num_samples, 1])
  # Set up fixed effects.
  mu = tf.get_variable("mu", [num_genes])

  sigma_ind = tf.sqrt(tf.exp(tf.get_variable("sigma_ind", [num_genes])))

  sigma_resid = tf.sqrt(tf.exp(tf.get_variable("sigma_resid", [num_genes])))


  # Set up random effects
  eta_ind = Normal(loc=tf.zeros([num_individuals, num_genes]), scale= tf.matmul(tf.ones([num_individuals,num_genes]),tf.matrix_diag(sigma_ind)))

  # Set up factors
  #U = Normal(loc=0.0, scale=1, sample_shape=[num_samples, K])
  #V = Normal(loc=0.0, scale=1.0, sample_shape=[K, num_genes])
  #U = tf.exp(tf.get_variable("U", [num_samples, K]))
  
  # higher values of sparsity parameter result in a more sparse solution
  sparsity_parameter = 1.0
  U = Gamma(concentration=1.0, rate=sparsity_parameter, sample_shape=[num_samples,K])
  V = tf.get_variable("V", [K, num_genes])

  yhat = (tf.matmul(U, V) + tf.gather_nd(eta_ind, ind_ph) + tf.matmul(tf.ones([num_samples, num_genes]), tf.matrix_diag(mu)))
  y = Normal(loc=yhat, scale=tf.matmul(tf.ones([num_samples, num_genes]), tf.matrix_diag(sigma_resid)))

  ###############################
  ## Inference set up
  ###############################
  q_ind_s = Normal(
    loc=tf.get_variable("q_ind_s/loc", [num_individuals, num_genes]),
    scale=tf.nn.softplus(tf.get_variable("q_ind_s/scale", [num_individuals, num_genes])))

  qU = lognormal_q(U.shape)

  #qU = Normal(loc=tf.get_variable("qU/loc", [num_samples, K]),
   #           scale=tf.nn.softplus(
   #               tf.get_variable("qU/scale", [num_samples, K])))
  #qV = Normal(loc=tf.get_variable("qV/loc", [K, num_genes]),
   #           scale=tf.nn.softplus(
    #              tf.get_variable("qV/scale", [K, num_genes])))

  latent_vars = {
    U: qU,
    eta_ind: q_ind_s}
  data = {
    y: Y_train,
    ind_ph: Z_train}
  inference = ed.KLqp(latent_vars, data)
  tf.global_variables_initializer().run()
  inference.run(n_iter=500)
  # qU.distribution.scale.eval()
  pdb.set_trace()





##################################
# Run bayesian linear regression
# http://edwardlib.org/tutorials/supervised-regression
##################################
#bayesian_linear_regression()


##################################
# Run probabilistic matrix factorization
# https://github.com/blei-lab/edward/blob/master/examples/probabilistic_matrix_factorization.py
##################################
#bayesian_matrix_factorization()



##################################
# Run bayesian LMM
#http://edwardlib.org/tutorials/linear-mixed-effects-models
##################################
# bayesian_linear_mixed_model()


##################################
# Run multivariate bayesian LMM
#http://edwardlib.org/tutorials/linear-mixed-effects-models
##################################
# multivariate_bayesian_linear_mixed_model()


##################################
# Run multivariate bayesian LMM and matrix factorization
##################################
multivariate_bayesian_linear_mixed_model_and_factorization()