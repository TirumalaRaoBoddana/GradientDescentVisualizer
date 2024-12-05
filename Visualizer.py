import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import GDregressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
st.sidebar.title("Gradient Descent Visualization")
learning_rate = 0.001
if 'epochs' not in st.session_state:
  st.session_state['epochs']=2
if 'cost_history' not in st.session_state:
  st.session_state.cost_history=[]
if 'epoch_history' not in st.session_state:
  st.session_state.epoch_history=[]
if 'slope_history' not in st.session_state:
  st.session_state.slope_history=[]
if 'intercept_history' not in st.session_state:
  st.session_state.intercept_history=[]
st.subheader("Gradient Descent: ")
st.markdown(
    """
    Gradient Descent is a fundamental optimization algorithm in machine learning used to minimize a loss (or cost) function. By iteratively adjusting the parameters of a model, gradient descent finds the optimal values that minimize the error between predictions and actual data.
    <div>
        <div>
          <h4>
            Illustration:
          </h4>
          <p>
            Imagine you're hiking in a hilly terrain, and you want to reach the lowest point (global minimum). Gradient Descent works as follows:
          </p>
          <ol>
            <li>It looks at the slope of the terrain (gradient of the cost function).</li>
            <li>Takes a step downhill in the direction of steepest descent.</li>
            <li>Repeats until it reaches the lowest point or gets close enough.</li>
          </ol>
        </div>
        <h4>How Gradient Descent Works?</h4>
        <ol>
          <li>
            <b>Objective: Minimize the Cost Function</b>
            <p>
                The cost function quantifies the error between the predicted values and the actual values of the target variable. For instance:
                <ul>
                    <li>
                        In simple linear regression, the cost function is the Mean Squared Error (MSE)
                    </li>
                    <li>In logistic regression, it’s the log-loss</li>
                </ul>
            </p>
          </li>
          <li>
              <b>Finding Optimal Parameters:</b>
              <p>Gradient Descent iteratively updates the parameters (e.g., weights in linear regression or coefficients in a neural network) to move closer to the global minimum of the cost function.</p>
          </li>
          <li>
            <b>Updation Rule: </b><br>
            <center>
                <b style='font-size:30px;'>W<sub>new</sub> = W<sub>old</sub> - &eta; (∂L/∂w)</b>
            </center>
            <div>
              Where <br>
              <ul>
                <li>W is the weight vector, W = [w<sub>0</sub>, w<sub>1</sub>, w<sub>3</sub>, ..... w<sub>n</sub>]</li>
                <li>W<sub>new</sub> is the updated weight vector</li>
                <li>W<sub>old</sub> is the old weight vector</li>
                <li>L is the loss function or cost function.</li>
                <li>(∂L/∂w) is the partial derivative of the loss function w.r.t weight vector (Gradient of the loss function). It shows the direction of steepest descent</li>
                <li>η is the learning rate, usually 0.001, 0.01, 0.1, etc.</li>
              </ul>
            </div>
          </li>
          <li>
            <b>Learning Rate (η)</b>
            <p>
              The learning rate (η) is a hyperparameter in Gradient Descent that controls the size of the steps the algorithm takes towards minimizing the cost function. It plays a crucial role in determining the efficiency and success of the optimization process.
              <ul>
                <li>Small η:<br>
                  <ol>
                    <li>Steps are small.</li>
                    <li>Convergence is slow and can take a long time.</li>
                    <li>However, it’s less likely to overshoot the minimum.</li>
                  </ol>
                </li>
                <li>Large η:<br>
                  <ol>
                    <li>Steps are large.</li>
                    <li>Convergence may be faster if the learning rate is optimal.</li>
                    <li>If η is too large, it may overshoot the minimum or fail to converge, oscillating around the solution.</li>
                  </ol>
                </li>
              </ul>
            </p>
          </li>
        </ol>    
    </div>
    """,
    unsafe_allow_html=True
)
st.subheader("Gradient Descent in Linear Regression")
st.write("Linear Regression is one of the most fundamental and widely used algorithms in machine learning for predictive modeling. It is a type of supervised learning algorithm, meaning it learns from labeled data and makes predictions based on input features.")
st.write("Simple Linear regression aims to establish a relationship between the input variable x and the output variable Y by fitting a linear equation to the observed data. The equation for linear regression in a simple one-variable case is:")
st.markdown("""
            <center style="font-size:20px;">
                 Y = m x + b 
            </center><br>
            <ul>
              <li>
                  y is the dependent variable (target),
              </li>
              <li>
                x is the independent variable (feature),
              </li>
              <li>
                b is the intercept 
              </li>
              <li>
                   m is the coefficient (slope) of feature x 
              </li>
            </ul>
""", unsafe_allow_html=True)
st.subheader("Loss Function in Linear Regression:")
st.image("linearregression.webp")
st.markdown("The sum of squared errors can be given as  $$ E = d_1^2 + d_2^2 + d_3^2 + \dots + d_n^2 $$ where $$ d_i $$ is the difference between the actual value $$Y_i$$ and the predicted value $$\hat{Y}_i $$. In compact notation, this can be represented as:", unsafe_allow_html=True)
st.latex(r'''\sum_{i=1}^{n} d_i^2 \quad \text{where} \quad d_i = Y_i - \hat{Y}_i.
''')

st.markdown("The error can be represented as:", unsafe_allow_html=True)

st.latex(r'''E = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \quad \text{where} \quad \hat{y}_i = m x_i + b''')

st.latex(r'''
    E(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - m x_i - b)^2
''')

st.write("The error is a function of the coefficients of $$ x $$ (i.e., $$ m $$) and the intercept $$ b $$. We need to find the values of \( m \) and \( b \) such that the value of the loss function is minimized. Here comes the Gradient Descent algorithm.")

st.subheader("Gradient Descent Algorithm:")
st.markdown("""
      <ol>
        <li>
          Initialize the parameters m and b
        </li>
        <li>
          set learning rate(η)=0.01(according to the dataset)
        </li>
        <li>
          set epochs or iterations=1000(as your wish)    
        </li>
        <li>
            for i in range(1,epochs,1):
            <ul>
              <li>Compute the Gradients ∂E/∂m and ∂E/∂b at [m<sub>old</sub>,b<sub>old</sub>]</li>
              <li><b>Update the parameters:</b><br>
                <ul>
                  <li> m<sub>new</sub> = m<sub>old</sub> - η * ∂E/∂m</li>
                  <li>b<sub>new</sub> =  b<sub>old</sub>- η * ∂E/∂b</li>
                </ul>
              </li>
            </ul>
        </li>     
      </ol>
""",unsafe_allow_html=True)
st.subheader("calculating the Gradients: ")
###
st.latex(r'''
\text{Cost function: } E(m, b) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - (m x_i + b))^2
''')
st.divider()
st.markdown("""
  <h5><b>Differentiation of the Loss function wrt to m: </b></h5>
""",unsafe_allow_html=True)
st.latex(r'''
\frac{\partial E(m, b)}{\partial m} = \frac{\partial}{\partial m} \left( \frac{1}{2n} \sum_{i=1}^{n} (y_i - (m x_i + b))^2 \right)
''')

st.latex(r'''
\frac{\partial E(m, b)}{\partial m}= \frac{1}{2n} \sum_{i=1}^{n} 2(y_i - (m x_i + b)) \cdot \frac{\partial}{\partial m} \left( y_i - (m x_i + b) \right)
''')

st.latex(r'''
\frac{\partial E(m, b)}{\partial m}= -\frac{1}{n} \sum_{i=1}^{n} x_i \cdot (y_i - (m x_i + b))
''')
st.divider()
st.markdown("""
  <h5><b>Differentiation of the Loss function wrt to b: </b></h5>
""",unsafe_allow_html=True)
st.latex(r'''
\frac{\partial E(m, b)}{\partial b} = \frac{\partial}{\partial b} \left( \frac{1}{2n} \sum_{i=1}^{n} (y_i - (m x_i + b))^2 \right)
''')

st.latex(r'''
\frac{\partial E(m, b)}{\partial b}= -\frac{1}{n} \sum_{i=1}^{n} (y_i - (m x_i + b))
''')
st.divider()
st.write("These gradients tell us how much the cost function changes with respect to changes in $$ m $$ and $$ b $$.")
st.subheader("Dataset:")
st.write("Consider the dataset consisting of 1 input column(YearsExperience) and 1 output column(Salary)")
data=pd.read_csv("Salary_dataset.csv")
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0], data.iloc[:, 1], test_size=0.2, random_state=2)
col1,col2=st.columns(2,vertical_alignment='center')
with col1:
  st.dataframe(pd.DataFrame(data,columns=["YearsExperience","Salary"]),use_container_width=True)
with col2:
  st.write("The dataset consisting of the 30 employees experience and Salary.lets plot the data in 2d space..")
  #plotting the dataset
  plt.figure(figsize=(10, 6))
  plt.scatter(x_train, y_train, color='blue', label='Data points')
  plt.xlabel("YearsExperience")
  plt.ylabel("Salary")
  plt.title("YearsExperience vs Salary")
  plt.grid(True)
  plt.legend()
  st.pyplot(plt)
gd = GDregressor.GDRegressor(learning_rate=0.001, epochs=4000) 
gd.fit(pd.DataFrame(x_train), y_train)
lm=LinearRegression()
lm.fit(pd.DataFrame(x_train),y_train)
history=gd.history
def visualize(epochs):
  if(st.session_state.epochs>=4000):
    st.write("regression line is converged at 3999 iterations")
  else:
    if(st.session_state.epochs>=2):
      st.subheader("Visualization:")
    if(st.session_state.epochs<=1):
      st.session_state.epochs=2
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='blue', label='Data points')
    plt.plot(x_train,x_train*history[st.session_state.epochs-1][0]+history[st.session_state.epochs-1][1],color='green')
    plt.plot(x_train,x_train*history[st.session_state.epochs-2][0]+history[st.session_state.epochs-2][1],color='black',linestyle='dashdot')
    caption="regression line after {} iterations(slope={:.4f},intercept={:.4f})".format(st.session_state.epochs-2,history[st.session_state.epochs-2][0],history[st.session_state.epochs-2][1])
    plt.plot(x_train,x_train*lm.coef_+lm.intercept_,color='red')
    plt.xlabel("YearsExperience")
    plt.ylabel("Salary")
    plt.title("YearsExperience vs Salary")
    plt.grid(True)
    plt.legend(["data points","regression line after {} iterations(slope={:.4f},intercept={:.4f})".format(st.session_state.epochs-1,history[st.session_state.epochs-1][0],history[st.session_state.epochs-1][1]),caption,"regression line with (slope={:0.4f} and intercept={:.4f})".format(lm.coef_[0],lm.intercept_)])
    return st.pyplot(plt)
def calculation():
  if(st.session_state.epochs>=4000):
    st.write("regression line is converged at 3999 iterations")
  else:
    st.subheader("Calculations: ")
    m_curr = history[st.session_state.epochs - 1][0]
    b_curr = history[st.session_state.epochs - 1][1]
    learning_rate = 0.001
    n = len(x_train)
    y_pred = m_curr * x_train + b_curr
    gradient_m = (-2 / n) * sum((y_train - y_pred) * x_train)
    gradient_b = (-2 / n) * sum(y_train - y_pred)
    m_new = m_curr - learning_rate * gradient_m
    b_new = b_curr - learning_rate * gradient_b
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
                    <h6>For previous iteration:</h6>
                    Slope (m): {history[st.session_state.epochs - 2][0]:.4f}<br>
                    Intercept (b): {history[st.session_state.epochs - 2][1]:.4f}
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
                    <h6>For present iteration:</h6>
                    Slope (m): {m_curr:.4f}<br>
                    Intercept (b): {b_curr:.4f}<br>
                    Gradient Slope: {gradient_m:.4f}<br>
                    Gradient Intercept: {gradient_b:.4f}
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
                    <h6>For next iteration:</h6>
                    m<sub>new</sub>=m<sub>old</sub>-η*Gradient Slope<br>
                    m<sub>new</sub>={m_curr:.4f}-({0.001*gradient_m:.4f})
                    <mark style='background-color:yellow;'><b>m<sub>new</sub>={m_new:.4f}</b></mark><br>
                    b<sub>new</sub>=b<sub>old</sub>-η*Gradient Intercept<br>
                    b<sub>new</sub>={b_curr:.4f}-{0.001*gradient_b:.4f}<br>
                    <mark style='background-color:yellow;'><b>b<sub>new</sub>={b_new:.4f}</b></mark><br>
        """, unsafe_allow_html=True)
def visualize_cost(x_train,y_train,history,epochs):
  if(epochs>=4000):
    st.write("regression line is converged at 3999 iterations")
  else:
    def get_cost(x_train, y_train, history,epochs):
      predictions = history[epochs][0] * x_train + history[epochs][1]  
      errors = y_train - predictions  
      cost = (1 / (2 * len(x_train))) * np.sum(errors ** 2)   
      return cost
    st.session_state.epoch_history.append(epochs)
    st.session_state.cost_history.append(get_cost(x_train,y_train,history,epochs))
    plt.figure(figsize=(8, 6))
    plt.plot(st.session_state.epoch_history,st.session_state.cost_history,color="blue")
    plt.scatter(st.session_state.epoch_history,st.session_state.cost_history,color="red",label="costs")
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Convergence of Cost') 
    plt.legend()
    return st.pyplot(plt)
def visualize_slope(history,epochs):
  st.session_state.slope_history.append(history[epochs][0])
  plt.figure(figsize=(8, 6))
  plt.plot(st.session_state.epoch_history,st.session_state.slope_history,color="blue")
  plt.scatter(st.session_state.epoch_history,st.session_state.slope_history,color="red",label="slopes")
  plt.xlabel('Iteration')
  plt.ylabel('Slope')
  plt.title('Convergence of Slope') 
  plt.legend()
  return st.pyplot(plt)
def visualize_intercept(history,epochs):
  st.session_state.intercept_history.append(history[epochs][1])
  plt.figure(figsize=(8, 6))
  plt.plot(st.session_state.epoch_history,st.session_state.intercept_history,color="blue")
  plt.scatter(st.session_state.epoch_history,st.session_state.intercept_history,color="red",label="slopes")
  plt.xlabel('Iteration')
  plt.ylabel('Intercept')
  plt.title('Convergence of Intercept') 
  plt.legend()
  return st.pyplot(plt)
if st.sidebar.button("Next Iteration"):
  visualize(st.session_state.epochs)  
  calculation()
  visualize_cost(x_train,y_train,history,st.session_state.epochs)
  visualize_slope(history,st.session_state.epochs)
  visualize_intercept(history,st.session_state.epochs)
  st.session_state.epochs+=1
st.sidebar.write("For faster convergence increment iteration by 100") 
st.subheader("Animation of gradient Descent")
st.image("gradient_descent(slope known).gif",caption="convergence of intercept when slope is known")
st.image("gradient_descent(intercept known).gif",caption="convergence of slope when intercept is known")
st.image("gradient_descent(both_unknown).gif",caption="convergence when both are unknown")
