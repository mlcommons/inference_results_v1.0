/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*! \file
  \brief Functor performing scaling operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies channel based scaling.
///
/// D = accumulator * source
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class ChannelBasedScaling {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() { }
  };
private:

  //
  // Data members
  //

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  ChannelBasedScaling(Params const &params) { }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return true; //beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    //if (k_partition) {
    //  beta_ = ElementCompute(1);
    //}
  }
  
  /// Computes linear scaling: D = accumulator * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator, 
    FragmentOutput const &source) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations

    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_acc_source;

    intermediate = mul_acc_source(converted_accumulator, converted_source);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }
  /// This method should not be called because we always need scales, but is required for compilation
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations

    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_accumulator;

    intermediate = mul_accumulator(ElementCompute(1), converted_accumulator);   // never happens

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass
