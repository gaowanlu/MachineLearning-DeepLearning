## TensorFlow Functions
def cube(x):
    return x ** 3


cube(2)
cube(tf.constant(2.0))
tf_cube = tf.function(cube)
tf_cube
tf_cube(2)
tf_cube(tf.constant(2.0))
# TF Functions and Concrete Functions
concrete_function = tf_cube.get_concrete_function(tf.constant(2.0))
concrete_function.graph
concrete_function(tf.constant(2.0))
concrete_function is tf_cube.get_concrete_function(tf.constant(2.0))
### Exploring Function Definitions and Graphs

concrete_function.graph
ops = concrete_function.graph.get_operations()
ops
pow_op = ops[2]
list(pow_op.inputs)
pow_op.outputs
concrete_function.graph.get_operation_by_name('x')
concrete_function.graph.get_tensor_by_name('Identity:0')
concrete_function.function_def.signature


### How TF Functions Trace Python Functions to Extract Their Computation Graphs

@tf.function
def tf_cube(x):
    print("print:", x)
    return x ** 3


result = tf_cube(tf.constant(2.0))
result
result = tf_cube(2)
result = tf_cube(3)
result = tf_cube(tf.constant([[1., 2.]]))  # New shape: trace!
result = tf_cube(tf.constant([[3., 4.], [5., 6.]]))  # New shape: trace!
result = tf_cube(tf.constant([[7., 8.], [9., 10.], [11., 12.]]))  # no trace

#It is also possible to specify a particular input signature:


@tf.function(input_signature=[tf.TensorSpec([None, 28, 28], tf.float32)])
def shrink(images):
    print("Tracing", images)
    return images[:, ::2, ::2]  # drop half the rows and columns


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
img_batch_1 = tf.random.uniform(shape=[100, 28, 28])
img_batch_2 = tf.random.uniform(shape=[50, 28, 28])
preprocessed_images = shrink(img_batch_1)  # Traces the function.
preprocessed_images = shrink(img_batch_2)  # Reuses the same concrete function.
img_batch_3 = tf.random.uniform(shape=[2, 2, 2])
try:
    # rejects unexpected types or shapes
    preprocessed_images = shrink(img_batch_3)
except ValueError as ex:
    print(ex)


### Using Autograph To Capture Control Flow
@tf.function
def add_10(x):
    for i in range(10):
        x += 1
    return x


add_10(tf.constant(5))
add_10.get_concrete_function(tf.constant(5)).graph.get_operations()

#A "dynamic" loop using `tf.while_loop()`:


@tf.function
def add_10(x):
    def condition(i, x): return tf.less(i, 10)
    def body(i, x): return (tf.add(i, 1), tf.add(x, 1))
    final_i, final_x = tf.while_loop(condition, body, [tf.constant(0), x])
    return final_x


add_10(tf.constant(5))
add_10.get_concrete_function(tf.constant(5)).graph.get_operations()

#A "dynamic" `for` loop using `tf.range()` (captured by autograph):


@tf.function
def add_10(x):
    for i in tf.range(10):
        x = x + 1
    return x


add_10.get_concrete_function(tf.constant(0)).graph.get_operations()

### Handling Variables and Other Resources in TF Functions
counter = tf.Variable(0)


@tf.function
def increment(counter, c=1):
    return counter.assign_add(c)


increment(counter)
increment(counter)
function_def = increment.get_concrete_function(counter).function_def
function_def.signature.input_arg[0]
counter = tf.Variable(0)


@tf.function
def increment(c=1):
    return counter.assign_add(c)


increment()
increment()
function_def = increment.get_concrete_function().function_def
function_def.signature.input_arg[0]


class Counter:
    def __init__(self):
        self.counter = tf.Variable(0)

    @tf.function
    def increment(self, c=1):
        return self.counter.assign_add(c)


c = Counter()
c.increment()
c.increment()


@tf.function
def add_10(x):
    for i in tf.range(10):
        x += 1
    return x


tf.autograph.to_code(add_10.python_function)


def display_tf_code(func):
    from IPython.display import display, Markdown
    if hasattr(func, "python_function"):
        func = func.python_function
    code = tf.autograph.to_code(func)
    display(Markdown('```python\n{}\n```'.format(code)))


display_tf_code(add_10)


## Using TF Functions with tf.keras (or Not)
#By default, tf.keras will automatically convert your custom code into TF Functions, no need to use
#`tf.function()`:
# Custom loss function
def my_mse(y_true, y_pred):
    print("Tracing loss my_mse()")
    return tf.reduce_mean(tf.square(y_pred - y_true))
# Custom metric function


def my_mae(y_true, y_pred):
    print("Tracing metric my_mae()")
    return tf.reduce_mean(tf.abs(y_pred - y_true))
# Custom layer


class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.biases = self.add_weight(name='bias',
                                      shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True)
        super().build(input_shape)

    def call(self, X):
        print("Tracing MyDense.call()")
        return self.activation(X @ self.kernel + self.biases)


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
# Custom model


class MyModel(keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = MyDense(30, activation="relu")
        self.hidden2 = MyDense(30, activation="relu")
        self.output_ = MyDense(1)

    def call(self, input):
        print("Tracing MyModel.call()")
        hidden1 = self.hidden1(input)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input, hidden2])
        output = self.output_(concat)
        return output


model = MyModel()
model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae])
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
model.evaluate(X_test_scaled, y_test)
#You can turn this off by creating the model with `dynamic = True` (or calling `super().__init__(dynamic=True, **kwargs)` in the model's constructor):
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = MyModel(dynamic=True)
model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae])

#Not the custom code will be called at each iteration. Let's fit, validate and evaluate with tiny datasets to avoid getting too much output:
model.fit(X_train_scaled[:64], y_train[:64], epochs=1,
          validation_data=(X_valid_scaled[:64], y_valid[:64]), verbose=0)
model.evaluate(X_test_scaled[:64], y_test[:64], verbose=0)

#Alternatively, you can compile a model with `run_eagerly=True`:
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = MyModel()
model.compile(loss=my_mse, optimizer="nadam",
              metrics=[my_mae], run_eagerly=True)
model.fit(X_train_scaled[:64], y_train[:64], epochs=1,
          validation_data=(X_valid_scaled[:64], y_valid[:64]), verbose=0)
model.evaluate(X_test_scaled[:64], y_test[:64], verbose=0)



# Custom Optimizers
#Defining custom optimizers is not very common, but in case you are one of the happy few who gets to write one, here is an example:
class MyMomentumOptimizer(keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.9, name="MyMomentumOptimizer", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get(
            "lr", learning_rate))  # handle lr=learning_rate
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("momentum", momentum)

    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "momentum")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)  # handle learning rate decay
        momentum_var = self.get_slot(var, "momentum")
        momentum_hyper = self._get_hyper("momentum", var_dtype)
        momentum_var.assign(momentum_var * momentum_hyper -
                            (1. - momentum_hyper) * grad)
        var.assign_add(momentum_var * lr_t)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
        }


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([keras.layers.Dense(1, input_shape=[8])])
model.compile(loss="mse", optimizer=MyMomentumOptimizer())
model.fit(X_train_scaled, y_train, epochs=5)
