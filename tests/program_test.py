# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from collections import namedtuple
from jax.program import kernel, like, Program

import jax.numpy as jnp
import logging
import unittest

logging.basicConfig(level=logging.DEBUG)

x = jnp.ones((3, 4), jnp.float32) * 4.0
b = jnp.ones((3, 4), jnp.float32)

Params = namedtuple("Params", "x,b")
params = Params(x, b)

class TrivialKernel(Program):
  _params = params
  _x = params.x

  def get_params(self):
    return self._params

  def run(self, multiplier=like(x)):
    result = self._linear(multiplier, self._params.x, self._params.b)
    self._x = result
    return result

  def set_params(self, new_params=like(params)):
    self._params = new_params

  @kernel
  def _linear(m, x, b):
    return m * x + b

class ProfilerTest(unittest.TestCase):
  def testIRGeneration(self):
    m = TrivialKernel()
    src = str(Program.get_mlir_module(m))
    self.assertTrue("module @trivial_kernel" in src)
    self.assertTrue("func @get_params" in src)
    self.assertTrue("func @set_params" in src)
    self.assertTrue("func @run" in src)
    self.assertTrue("ml_program.global_store" in src)
    self.assertTrue("ml_program.global_load" in src)

if __name__ == "__main__":
  absltest.main()

