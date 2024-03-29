/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace custom_helper_op {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

// TODO(qyu): Move this to core/framework/common_shape_fns.h
Status CostVolumeShapeFn(InferenceContext *c) {
  ShapeHandle images_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &images_shape));
  ShapeHandle homos_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &homos_shape));
  auto batch_dim = c->Dim(images_shape, 0);
  auto height    = c->Dim(images_shape, 2);
  auto width     = c->Dim(images_shape, 3);
  auto channel_dim = c->Dim(images_shape, 4);
  auto depth_dim = c->Dim(homos_shape, 2);
  c->set_output(0, c->MakeShape({batch_dim, height, width, depth_dim, channel_dim}));
  c->set_output(1, c->MakeShape({batch_dim, height, width, depth_dim, 1}));
  return Status::OK();
}

// TODO(qyu): Move this to core/framework/common_shape_fns.h
Status CostAggregateShapeFn(InferenceContext *c) {
  ShapeHandle ref_image_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &ref_image_shape));
  ShapeHandle offsets_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &offsets_shape));
  auto batch_dim = c->Dim(ref_image_shape, 0);
  auto height    = c->Dim(ref_image_shape, 1);
  auto width     = c->Dim(ref_image_shape, 2);
  auto depth_dim = c->Dim(offsets_shape, 1);
  c->set_output(0, c->MakeShape({batch_dim, height, width, depth_dim}));
  c->set_output(1, c->MakeShape({batch_dim, height, width, depth_dim}));
  return Status::OK();
}

// TODO(qyu): Move this to core/framework/common_shape_fns.h
Status SparseConv2DShapeFn(InferenceContext *c) {
  ShapeHandle image_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &image_shape));
  ShapeHandle filter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));
  auto batch_dim = c->Dim(image_shape, 0);
  auto height    = c->Dim(image_shape, 1);
  auto width     = c->Dim(image_shape, 2);
  auto out_channels = c->Dim(filter_shape, 0);
  c->set_output(0, c->MakeShape({batch_dim, height, width, out_channels}));
  return Status::OK();
}

static const char kCostVolumeDoc[] = R"doc(
Applies the given transform to each of the images.

Input `image` is a `Tensor` in NHWC format (where the axes are image in batch,
rows, columns, and channels. Input `transforms` is a num_images x 8 or 1 x 8
matrix, where each row corresponds to a 3 x 3 projective transformation matrix,
with the last entry assumed to be 1. If there is one row, the same
transformation will be applied to all images.

If one row of `transforms` is `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps
the *output* point `(x, y)` to a transformed *input* point
`(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
`k = c0 x + c1 y + 1`. If the transformed point lays outside of the input
image, the output pixel is set to 0.

images: 4D `Tensor`, input image(s) in NHWC format.
transforms: 2D `Tensor`, projective transform(s) to apply to the image(s).

transformed_images: 4D `Tensor`, image(s) in NHWC format, generated by applying
the `transforms` to the `images`. Satisfies the description above.
)doc";

static const char kCostVolumeGradDoc[] = R"doc(
Applies the given transform to each of the images.

Input `image` is a `Tensor` in NHWC format (where the axes are image in batch,
rows, columns, and channels. Input `transforms` is a num_images x 8 or 1 x 8
matrix, where each row corresponds to a 3 x 3 projective transformation matrix,
with the last entry assumed to be 1. If there is one row, the same
transformation will be applied to all images.

If one row of `transforms` is `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps
the *output* point `(x, y)` to a transformed *input* point
`(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
`k = c0 x + c1 y + 1`. If the transformed point lays outside of the input
image, the output pixel is set to 0.

images: 4D `Tensor`, input image(s) in NHWC format.
transforms: 2D `Tensor`, projective transform(s) to apply to the image(s).

grad: 4D `Tensor`, image(s) in NHWC format, generated by applying
the `transforms` to the `images`. Satisfies the description above.

image_grad: 4D `Tensor`, image(s) in NHWC format, generated by applying
the `transforms` to the `images`. Satisfies the description above.
)doc";

static const char kCostAggregateDoc[] = R"doc(
Applies the given transform to each of the images.

Input `image` is a `Tensor` in NHWC format (where the axes are image in batch,
rows, columns, and channels. Input `transforms` is a num_images x 8 or 1 x 8
matrix, where each row corresponds to a 3 x 3 projective transformation matrix,
with the last entry assumed to be 1. If there is one row, the same
transformation will be applied to all images.

If one row of `transforms` is `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps
the *output* point `(x, y)` to a transformed *input* point
`(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
`k = c0 x + c1 y + 1`. If the transformed point lays outside of the input
image, the output pixel is set to 0.

ref_image: 4D `Tensor`, input image(s) in NHWC format.
src_images: 5D `Tensor`, projective transform(s) to apply to the image(s).
base_plane: 4D `Tensor`, input image(s) in NHWC format.
offsets: 2D `Tensor`, projective transform(s) to apply to the image(s).
rs: 4D `Tensor`, input image(s) in NHWC format.
ts: 3D `Tensor`, projective transform(s) to apply to the image(s).

cost: 4D `Tensor`, image(s) in NHWC format, generated by applying
the `transforms` to the `images`. Satisfies the description above.
cost_mask: 4D `Tensor`, image(s) in NHWC format, generated by applying
the `transforms` to the `images`. Satisfies the description above.
)doc";

static const char kCostAggregateGradDoc[] = R"doc(
Applies the given transform to each of the images.

Input `image` is a `Tensor` in NHWC format (where the axes are image in batch,
rows, columns, and channels. Input `transforms` is a num_images x 8 or 1 x 8
matrix, where each row corresponds to a 3 x 3 projective transformation matrix,
with the last entry assumed to be 1. If there is one row, the same
transformation will be applied to all images.

If one row of `transforms` is `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps
the *output* point `(x, y)` to a transformed *input* point
`(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
`k = c0 x + c1 y + 1`. If the transformed point lays outside of the input
image, the output pixel is set to 0.

ref_image: 4D `Tensor`, input image(s) in NHWC format.
src_images: 5D `Tensor`, projective transform(s) to apply to the image(s).
base_plane: 4D `Tensor`, input image(s) in NHWC format.
offsets: 2D `Tensor`, projective transform(s) to apply to the image(s).
rs: 4D `Tensor`, input image(s) in NHWC format.
ts: 3D `Tensor`, projective transform(s) to apply to the image(s).
cost_grad: 4D `Tensor`, image(s) in NHWC format, generated by applying
the `transforms` to the `images`. Satisfies the description above.
cost_mask: 4D `Tensor`, image(s) in NHWC format, generated by applying
the `transforms` to the `images`. Satisfies the description above.

ref_image_grad: 4D `Tensor`, input image(s) in NHWC format.
src_images_grad: 5D `Tensor`, projective transform(s) to apply to the image(s).
base_plane_grad: 4D `Tensor`, input image(s) in NHWC format.
)doc";

static const char kIndexInitializerDoc[] = R"doc(
fill  matrix with index.

output_shape: 1D `Tensor`, ouput data shape.
)doc";

}  // namespace

// V2 op supports output_shape.
// V2 op supports output_shape.
REGISTER_OP("CostVolume")
    .Input("ref_image: dtype")
    .Input("src_images: dtype")
    .Input("base_plane: dtype")
    .Input("offsets: dtype")
    .Input("rs: dtype")
    .Input("ts: dtype")
    .Attr("dtype: {float32,float64}")
    .Attr("reduce_method: string")
    .Attr("half_centor: bool")
    .Output("cost: dtype")
    .Output("cost_mask: int32")
    .SetShapeFn([](InferenceContext *c) {
      string reduce_method;
      TF_RETURN_IF_ERROR(c->GetAttr("reduce_method", &reduce_method));
      ShapeHandle ref_image_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &ref_image_shape));
      ShapeHandle offsets_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &offsets_shape));
      auto batch_dim = c->Dim(ref_image_shape, 0);
      auto height    = c->Dim(ref_image_shape, 1);
      auto width     = c->Dim(ref_image_shape, 2);
      auto channels  = c->Dim(ref_image_shape, 3);
      auto depth_dim = c->Dim(offsets_shape, 1);
      c->set_output(0, c->MakeShape({batch_dim, height, width, depth_dim, channels}));
      if(reduce_method == "MEAN"){
        c->set_output(1, c->MakeShape({batch_dim, height, width, depth_dim, 1}));
      } else {
        c->set_output(1, c->MakeShape({batch_dim, height, width, depth_dim, channels}));
      }

      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("CostVolumeGrad")
    .Input("ref_image: dtype")
    .Input("src_images: dtype")
    .Input("base_plane: dtype")
    .Input("offsets: dtype")
    .Input("rs: dtype")
    .Input("ts: dtype")
    .Input("cost_grad: dtype")
    .Input("cost_mask: int32")
    .Attr("dtype: {float32,float64}")
    .Attr("reduce_method: string")
    .Attr("half_centor: bool")
    .Output("ref_image_grad: dtype")
    .Output("src_images_grad: dtype")
    .Output("base_plane_grad: dtype")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      return Status::OK();
    });

// V2 op supports output_shape.
// V2 op supports output_shape.
REGISTER_OP("CostVolumeV2")
    .Input("ref_image: dtype")
    .Input("src_images: dtype")
    .Input("depth_grid: dtype")
    .Input("rs: dtype")
    .Input("ts: dtype")
    .Attr("dtype: {float32,float64}")
    .Attr("reduce_method: string")
    .Attr("groups: int")
    .Attr("half_centor: bool")
    .Output("cost: dtype")
    .Output("cost_mask: int32")
    .SetShapeFn([](InferenceContext *c) {
      string reduce_method;
      TF_RETURN_IF_ERROR(c->GetAttr("reduce_method", &reduce_method));
      int32 groups;
      TF_RETURN_IF_ERROR(c->GetAttr("groups", &groups));
      ShapeHandle ref_image_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &ref_image_shape));
      ShapeHandle depth_grid_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &depth_grid_shape));
      auto batch_dim = c->Dim(ref_image_shape, 0);
      auto height    = c->Dim(ref_image_shape, 1);
      auto width     = c->Dim(ref_image_shape, 2);
      auto channels  = c->Dim(ref_image_shape, 3);
      auto depth_dim = c->Dim(depth_grid_shape, 3);
      // auto channels_value = c->Value(channels);
      c->set_output(0, c->MakeShape({batch_dim, height, width, depth_dim, groups}));
      if(reduce_method == "MEAN"){
        c->set_output(1, c->MakeShape({batch_dim, height, width, depth_dim, 1}));
      } else {
        c->set_output(1, c->MakeShape({batch_dim, height, width, depth_dim, groups}));
      }

      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("CostVolumeGradV2")
    .Input("ref_image: dtype")
    .Input("src_images: dtype")
    .Input("depth_grid: dtype")
    .Input("rs: dtype")
    .Input("ts: dtype")
    .Input("cost_grad: dtype")
    .Input("cost_mask: int32")
    .Attr("dtype: {float32,float64}")
    .Attr("reduce_method: string")
    .Attr("groups: int")
    .Attr("half_centor: bool")
    .Output("ref_image_grad: dtype")
    .Output("src_images_grad: dtype")
    .Output("base_plane_grad: dtype")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      return Status::OK();
    });

// V2 op supports output_shape.
// V2 op supports output_shape.
REGISTER_OP("CostVolumeV3")
    .Input("ref_image: dtype")
    .Input("src_images: dtype")
    .Input("base_plane: dtype")
    .Input("offsets: dtype")
    .Input("rs: dtype")
    .Input("ts: dtype")
    .Attr("dtype: {float32,float64}")
    .Attr("reduce_method: string")
    .Attr("groups: int")
    .Attr("half_centor: bool")
    .Output("cost: dtype")
    .Output("cost_mask: int32")
    .SetShapeFn([](InferenceContext *c) {
      string reduce_method;
      TF_RETURN_IF_ERROR(c->GetAttr("reduce_method", &reduce_method));
      int32 groups;
      TF_RETURN_IF_ERROR(c->GetAttr("groups", &groups));
      ShapeHandle ref_image_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &ref_image_shape));
      ShapeHandle offsets_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &offsets_shape));
      auto batch_dim = c->Dim(ref_image_shape, 0);
      auto height    = c->Dim(ref_image_shape, 1);
      auto width     = c->Dim(ref_image_shape, 2);
      auto channels  = c->Dim(ref_image_shape, 3);
      auto depth_dim = c->Dim(offsets_shape, 1);
      // auto channels_value = c->Value(channels);
      c->set_output(0, c->MakeShape({batch_dim, height, width, depth_dim, 1}));
      if(reduce_method == "MEAN"){
        c->set_output(1, c->MakeShape({batch_dim, height, width, depth_dim, 1}));
      } else {
        c->set_output(1, c->MakeShape({batch_dim, height, width, depth_dim, 3}));
      }

      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("CostVolumeGradV3")
    .Input("ref_image: dtype")
    .Input("src_images: dtype")
    .Input("base_plane: dtype")
    .Input("offsets: dtype")
    .Input("rs: dtype")
    .Input("ts: dtype")
    .Input("cost_grad: dtype")
    .Input("cost_mask: int32")
    .Attr("dtype: {float32,float64}")
    .Attr("reduce_method: string")
    .Attr("groups: int")
    .Attr("half_centor: bool")
    .Output("ref_image_grad: dtype")
    .Output("src_images_grad: dtype")
    .Output("base_plane_grad: dtype")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("CostAggregate")
    .Input("ref_image: dtype")
    .Input("src_images: dtype")
    .Input("base_plane: dtype")
    .Input("offsets: dtype")
    .Input("rs: dtype")
    .Input("ts: dtype")
    .Attr("dtype: {float32,float64}")
    .Attr("reduce_method: string")
    .Attr("half_centor: bool")
    .Output("cost: dtype")
    .Output("cost_mask: int32")
    .SetShapeFn(CostAggregateShapeFn)
    .Doc(kCostAggregateDoc);

// V2 op supports output_shape.
REGISTER_OP("CostAggregateGrad")
    .Input("ref_image: dtype")
    .Input("src_images: dtype")
    .Input("base_plane: dtype")
    .Input("offsets: dtype")
    .Input("rs: dtype")
    .Input("ts: dtype")
    .Input("cost_grad: dtype")
    .Input("cost_mask: int32")
    .Attr("dtype: {float32,float64}")
    .Attr("reduce_method: string")
    .Attr("half_centor: bool")
    .Output("ref_image_grad: dtype")
    .Output("src_images_grad: dtype")
    .Output("base_plane_grad: dtype")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      return Status::OK();
    })
    .Doc(kCostAggregateGradDoc);

// V2 op supports output_shape.
REGISTER_OP("FeatureAggregate")
    .Input("src_images: dtype")
    .Input("base_plane: dtype")
    .Input("offsets: dtype")
    .Input("rs: dtype")
    .Input("ts: dtype")
    .Attr("dtype: {float32,float64}")
    .Attr("half_centor: bool")
    .Output("mapped_feature: dtype")
    .Output("mapped_mask: int32")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle src_image_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &src_image_shape));
      ShapeHandle base_plane_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &base_plane_shape));
      ShapeHandle offsets_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &offsets_shape));
      auto batch_dim = c->Dim(src_image_shape, 0);
      auto image_num = c->Dim(src_image_shape, 1);
      auto output_height    = c->Dim(base_plane_shape, 1);
      auto output_width     = c->Dim(base_plane_shape, 2);
      auto channels  = c->Dim(src_image_shape, 4);
      auto output_depth     = c->Dim(offsets_shape, 1);
      c->set_output(0, c->MakeShape({batch_dim, output_height, output_width, output_depth, image_num, channels}));
      c->set_output(1, c->MakeShape({batch_dim, output_height, output_width, output_depth, image_num, 1}));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("FeatureAggregateGrad")
    .Input("src_images: dtype")
    .Input("base_plane: dtype")
    .Input("offsets: dtype")
    .Input("rs: dtype")
    .Input("ts: dtype")
    .Input("mapped_feature_grad: dtype")
    .Input("mapped_mask: int32")
    .Attr("dtype: {float32,float64}")
    .Attr("half_centor: bool")
    .Output("src_images_grad: dtype")
    .Output("base_plane_grad: dtype")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("SparseConv2D")
    .Input("images: dtype")
    .Input("filter: dtype")
    .Input("base_plane: dtype")
    .Input("default_value: dtype")
    .Input("offsets: dtype")
    .Output("output: dtype")
    .Attr("dtype: {float, double}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(SparseConv2DShapeFn);

// V2 op supports output_shape.
REGISTER_OP("SparseConv2DGrad")
    .Input("images: dtype")
    .Input("filter: dtype")
    .Input("base_plane: dtype")
    .Input("default_value: dtype")
    .Input("offsets: dtype")
    .Input("out_grad: dtype")
    .Output("images_grad: dtype")
    .Output("filter_grad: dtype")
    .Output("base_plane_grad: dtype")
    .Output("default_value_grad: dtype")
    .Attr("dtype: {float, double}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      c->set_output(3, c->input(3));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("SparsePad")
    .Input("images: dtype")
    .Input("base_plane: int32")
    .Output("output: dtype")
    .Attr("dtype: {float, double}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int)")
    .SetShapeFn([](InferenceContext *c) {
      std::vector<int32> strides;
      TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
      ShapeHandle image_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &image_shape));
      auto batch_dim       = c->Dim(image_shape, 0);
      auto image_height    = c->Value(c->Dim(image_shape, 1));
      auto image_width     = c->Value(c->Dim(image_shape, 2));
      auto image_depth     = c->Value(c->Dim(image_shape, 3));
      auto image_channel   = c->Value(c->Dim(image_shape, 4));

      const auto out_height = image_height * strides[0];
      const auto out_width = image_width * strides[1];
      const auto out_depth = image_depth * strides[2];
      c->set_output(0, c->MakeShape({batch_dim, out_height, out_width, out_depth, image_channel}));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("SparsePadGrad")
    .Input("images: dtype")
    .Input("base_plane: int32")
    .Input("out_grad: dtype")
    .Output("images_grad: dtype")
    .Attr("dtype: {float, double}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int)")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("SparseConv3D")
    .Input("images: dtype")
    .Input("filters: dtype")
    .Input("default_value: dtype")
    .Input("base_plane: int32")
    .Output("output: dtype")
    .Attr("dtype: {float, double}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int)")
    .Attr("dynamic_default: bool")
    .SetShapeFn([](InferenceContext *c) {
      std::vector<int32> strides;
      TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
      ShapeHandle image_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &image_shape));
      ShapeHandle filter_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &filter_shape));
      auto batch_dim = c->Dim(image_shape, 0);
      auto image_height    = c->Value(c->Dim(image_shape, 1));
      auto image_width     = c->Value(c->Dim(image_shape, 2));
      auto image_depth     = c->Value(c->Dim(image_shape, 3));
      auto out_channel_num = c->Value(c->Dim(filter_shape, 4));

      const auto out_height = (image_height + strides[0] - 1)/strides[0];
      const auto out_width = (image_width + strides[1] - 1)/strides[1];
      const auto out_depth = (image_depth + strides[2] - 1)/strides[2];
      c->set_output(0, c->MakeShape({batch_dim, out_height, out_width, out_depth, out_channel_num}));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("SparseConv3DGrad")
    .Input("images: dtype")
    .Input("filters: dtype")
    .Input("default_value: dtype")
    .Input("base_plane: int32")
    .Input("out_grad: dtype")
    .Output("images_grad: dtype")
    .Output("filter_grad: dtype")
    .Output("default_value_grad: dtype")
    .Attr("dtype: {float, double}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int)")
    .Attr("dynamic_default: bool")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("SparseConv3DFast")
    .Input("images: dtype")
    .Input("filters: dtype")
    .Input("default_value: dtype")
    .Input("base_plane: int32")
    .Output("output: dtype")
    .Attr("dtype: {float, double}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int)")
    .Attr("dynamic_default: bool")
    .Attr("data_format: { 'NDHWC' }")
    .SetShapeFn([](InferenceContext *c) {
      std::vector<int32> strides;
      TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
      ShapeHandle image_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &image_shape));
      ShapeHandle filter_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &filter_shape));
      auto batch_dim = c->Dim(image_shape, 0);
      auto image_height    = c->Value(c->Dim(image_shape, 1));
      auto image_width     = c->Value(c->Dim(image_shape, 2));
      auto image_depth     = c->Value(c->Dim(image_shape, 3));
      auto out_channel_num = c->Value(c->Dim(filter_shape, 4));

      const auto out_height = (image_height + strides[0] - 1)/strides[0];
      const auto out_width = (image_width + strides[1] - 1)/strides[1];
      const auto out_depth = (image_depth + strides[2] - 1)/strides[2];
      c->set_output(0, c->MakeShape({batch_dim, out_height, out_width, out_depth, out_channel_num}));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("SparseConv3DFastGrad")
    .Input("images: dtype")
    .Input("filters: dtype")
    .Input("default_value: dtype")
    .Input("base_plane: int32")
    .Input("out_grad: dtype")
    .Output("images_grad: dtype")
    .Output("filter_grad: dtype")
    .Output("default_value_grad: dtype")
    .Attr("dtype: {float, double}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int)")
    .Attr("dynamic_default: bool")
    .Attr("data_format: { 'NDHWC' }")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("SparseConv3DTransposeFast")
    .Input("images: dtype")
    .Input("filters: dtype")
    .Input("default_value: dtype")
    .Input("base_plane: int32")
    .Input("output_size: int32")
    .Output("output: dtype")
    .Attr("dtype: {float, double}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int)")
    .Attr("dynamic_default: bool")
    .Attr("data_format: { 'NDHWC' }")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle image_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &image_shape));
      ShapeHandle filter_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &filter_shape));
      auto batch_dim = c->Dim(image_shape, 0);
      // auto image_height    = c->Value(c->Dim(image_shape, 1));
      // auto image_width     = c->Value(c->Dim(image_shape, 2));
      // auto image_depth     = c->Value(c->Dim(image_shape, 3));
      // auto out_channel_num = c->Value(c->Dim(filter_shape, 4));

      auto input_channel_num = c->Dim(filter_shape, 3);

      // Get size values from the size tensor.
      const Tensor *size_tensor = c->input_tensor(4);
      DimensionHandle input_height;
      DimensionHandle input_width;
      DimensionHandle input_depth;
      if (size_tensor == nullptr) {
        input_height = c->UnknownDim();
        input_width = c->UnknownDim();
        input_depth = c->UnknownDim();
      } else {
        if (size_tensor->dtype() != DT_INT32) {
          return errors::InvalidArgument(
              "Bad size input type for SparseConv3DTransposeFast: Expected DT_INT32 "
              "but got ",
              DataTypeString(size_tensor->dtype()), " for input #", 0,
              " in ", c->DebugString());
        }
        auto vec = size_tensor->vec<int32>();
        input_height = c->MakeDim(vec(0));
        input_width = c->MakeDim(vec(1));
        input_depth = c->MakeDim(vec(2));
      }
      // std::cout << c->MakeShape({batch_dim, input_height, input_width, input_depth, input_channel_num}) << std::endl;
      c->set_output(0, c->MakeShape({batch_dim, input_height, input_width, input_depth, input_channel_num}));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("SparseConv3DTransposeFastGrad")
    .Input("images: dtype")
    .Input("filters: dtype")
    .Input("default_value: dtype")
    .Input("base_plane: int32")
    .Input("output_size: int32")
    .Input("out_grad: dtype")
    .Output("images_grad: dtype")
    .Output("filter_grad: dtype")
    .Attr("dtype: {float, double}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int)")
    .Attr("dynamic_default: bool")
    .Attr("data_format: { 'NDHWC' }")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return Status::OK();
    });

// V2 op supports output_shape.
REGISTER_OP("IndexInitializer")
    .Input("output_shape: int32")
    .Attr("half_centor: bool")
    .Attr("dtype: {float32,float64}")
    .Output("output: dtype")
    .SetShapeFn([](InferenceContext* c) {
      // Verify shape of size input.
      ShapeHandle size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &size));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 3, &unused));

      // Get size values from the size tensor.
      const Tensor *size_tensor = c->input_tensor(0);
      DimensionHandle width;
      DimensionHandle height;
      DimensionHandle channels;
      if (size_tensor == nullptr) {
        width = c->UnknownDim();
        height = c->UnknownDim();
        channels = c->MakeDim(3);
      } else {
        // TODO(petewarden) - Remove once we have constant evaluation in C++ only.
        if (size_tensor->dtype() != DT_INT32) {
          return errors::InvalidArgument(
              "Bad size input type for SetOutputToSizedImage: Expected DT_INT32 "
              "but got ",
              DataTypeString(size_tensor->dtype()), " for input #", 0,
              " in ", c->DebugString());
        }
        auto vec = size_tensor->vec<int32>();
        height = c->MakeDim(vec(0));
        width = c->MakeDim(vec(1));
        channels = c->MakeDim(vec(2));
        if (vec(2) != 3) {
          return errors::InvalidArgument(
              "Bad output_shape size for IndexInitializer: last dimension mast be 3"
              "but got ",
              vec(2));
        }
      }

      c->set_output(0, c->MakeShape({height, width, channels}));
      return Status::OK();
    })
    .Doc(kIndexInitializerDoc);

REGISTER_OP("DecodePfm")
    .Input("input: string")
    .Output("image: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(
          0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), 1}));
      return Status::OK();
    });

}  // end namespace addons
}  // namespace tensorflow