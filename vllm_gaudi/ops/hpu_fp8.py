import os
from typing import Callable, Optional

import torch
from torch.nn.parameter import Parameter
from vllm_gaudi import envs
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                        FusedMoeWeightScaleSupported)

from vllm.model_executor.parameter import ChannelQuantScaleParameter
from vllm.model_executor.layers.quantization import fp8
from vllm.model_executor.layers.quantization.fp8 import (Fp8LinearMethod as OrigFp8LinearMethod, Fp8MoEMethod,
                                                         Fp8Config)
import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi.extension.ops import (VllmMixtureOfExpertsOpFP8PerChannel, VllmMixtureOfExpertsOpFP8)


class Fp8LinearMethod(OrigFp8LinearMethod):

    def create_weights(self, *args, **kwargs) -> None:
        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if not self.block_quant:
                layer = kwargs.get('layer')
                output_partition_sizes = kwargs.get('output_partition_sizes')
                output_size_per_partition = sum(output_partition_sizes)
                scale = ChannelQuantScaleParameter(
                    data=torch.empty(output_size_per_partition,
                                        dtype=torch.float32),
                    output_dim=0,
                    weight_loader=kwargs.get('weight_loader'),
                )
                scale[:] = torch.finfo(torch.float32).min
                # override to be weight_scale_inv
                layer.register_parameter("weight_scale_inv", scale)
                layer.register_parameter("weight_scale", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.quant_config = self.quant_config
        if self.block_quant:
            layer = hpu_ops.fp8_block_linear_postprocess_weights(layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
            return
        elif self.quant_config.activation_scheme == "static":
            layer.input_scale = Parameter(layer.input_scale.max(),
                                          requires_grad=False)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return hpu_ops.apply_block_fp8_linear_hpu(
                input=x,
                layer=layer,
                block_size=self.quant_config.weight_block_size,
                bias=bias,
                do_unpad=True,
                force_channel_fp8=envs.VLLM_HPU_FORCE_CHANNEL_FP8,
            )
        elif self.quant_config.activation_scheme == "static":
            x_fp8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0/layer.input_scale, False, False, torch.float8_e4m3fn)[0]
            return torch.ops.hpu.fp8_gemm_v2(
                A=x_fp8,
                trans_A=False,
                B=layer.weight,
                trans_B=True,
                D=None,
                out_dtype=x.dtype,
                A_scale_inv=layer.input_scale,
                B_scale_inv=layer.weight_scale_inv,
                bias=bias,
                accumulate=False)
        weight_scale = layer.weight_scale.transpose(0, 1)
        input_scale = getattr(layer, 'input_scale', None)
        return hpu_ops.apply_fp8_linear_hpu(input=x,
                                            weight=layer.weight,
                                            weight_scale=weight_scale,
                                            input_scale=input_scale,
                                            bias=bias,
                                            trans_B=False)

    def dequant_fp8_weight(self, layer) -> torch.Tensor:
        if hasattr(layer, "updated_fp8_weight") and layer.updated_fp8_weight:
            return layer.weight
        dequant_weight = hpu_ops.dequant_block_fp8_weight_naive(
            layer.weight,
            layer.weight_scale_inv.data,
            self.quant_config.weight_block_size,
            original_M=layer.orig_M,
            original_N=layer.orig_N,
            do_unpad=True,
        )
        return dequant_weight


@CustomOp.register_oot(name='Fp8MoEMethod')
class HPUFp8MoEMethod(Fp8MoEMethod):

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(quant_config, layer)

        # Disable marlin
        self.use_marlin = False

        # disable DeepGemm support.
        self.allow_deep_gemm = False

        self.topk_indices_dtype = None
        self.fused_experts = None

        self.fused_experts: Optional[Callable] = None

        # Slicing the batched tokens for DynamicMoE to reduce the memory consumption
        self.moe_slice_length = int(os.environ.get("VLLM_MOE_SLICE_LENGTH", 128000))

    def create_weights(self, *args, **kwargs) -> None:
        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        kwargs['weight_loader'] = hpu_ops.synced_weight_loader(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)
        if not self.block_quant:
            layer = kwargs.get('layer')
            num_experts = kwargs.get('num_experts')
            hidden_size = kwargs.get('hidden_size')
            intermediate_size_per_partition = kwargs.get('intermediate_size_per_partition')
            w13_weight_scale = ChannelQuantScaleParameter(data=torch.ones(
                num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32), output_dim=0,
                                                    weight_loader=kwargs.get('weight_loader'))
            w2_weight_scale = ChannelQuantScaleParameter(data=torch.ones(
                num_experts, hidden_size, dtype=torch.float32), output_dim=0,
                                                    weight_loader=kwargs.get('weight_loader'))
            # override to be inverse
            layer.register_parameter("w13_weight_scale", None)
            layer.register_parameter("w2_weight_scale", None)
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            kwargs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
            )
            setattr(w13_weight_scale, "quant_method", FusedMoeWeightScaleSupported.CHANNEL.value)
            setattr(w2_weight_scale, "quant_method", FusedMoeWeightScaleSupported.CHANNEL.value)
            # WA
            def _load_single_value(self, param: torch.nn.Parameter,
                                   loaded_weight: torch.Tensor, expert_id: int):
                param_data = param.data

                # Input scales can be loaded directly and should be equal.
                param_data[expert_id] = loaded_weight.cpu()
            FusedMoE._load_single_value = _load_single_value

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.local_num_experts
        ep_shift = layer.ep_rank * num_experts

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        if self.block_quant and not envs.VLLM_HPU_FORCE_CHANNEL_FP8:
            layer.moe_op = VllmMixtureOfExpertsOpFP8(
                num_experts,
                experts_min,
                experts_max,
            )
        else:
            layer.moe_op = VllmMixtureOfExpertsOpFP8PerChannel(
                num_experts,
                experts_min,
                experts_max,
            )
        if self.block_quant:
            layer = hpu_ops.fp8_block_moe_prepare_weights(layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
        else:
            if self.quant_config.is_checkpoint_fp8_serialized and\
                self.quant_config.activation_scheme == "static":
                    layer.w13_input_scale = torch.nn.Parameter(
                        layer.w13_input_scale.max(), requires_grad=False)
                    layer.w2_input_scale = torch.nn.Parameter(
                        layer.w2_input_scale.max(), requires_grad=False)
                    num_experts = layer.w13_weight.shape[0]
                    self.w13_weight_list = [layer.w13_weight.data[i,...] for i in range(num_experts)]
                    self.w2_weight_list = [layer.w2_weight.data[i,...] for i in range(num_experts)]
                    self.w13_weight_scale_list = [layer.w13_weight_scale_inv.data[i,...] for i in range(num_experts)]
                    self.w2_weight_scale_list = [layer.w2_weight_scale_inv.data[i,...] for i in range(num_experts)]
                    self.w2_input_scale_list = [layer.w2_input_scale.data.unsqueeze(0).repeat(num_experts)[i] for i in range(num_experts)]
            else:
                layer = hpu_ops.fp8_channel_moe_prepare_weights(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        **kwargs,
    ) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if use_grouped_topk or custom_routing_function is not None:
            topk_weights, topk_ids, zero_expert_result = FusedMoE.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.view(*x.shape[:-1], -1)
        topk_weights = topk_weights.view(*x.shape[:-1], -1)
        if self.quant_config.activation_scheme == "static":
            x_scale = layer.w13_input_scale.data
            x_fp8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0/x_scale, False, False, torch.float8_e4m3fn)[0]
            topk_weights = topk_weights.to(torch.bfloat16)

            batched_tokens = x.shape[0]
            num_experts = layer.local_num_experts
            ep_shift = layer.ep_rank * num_experts

            if batched_tokens > self.moe_slice_length:
                final_hidden_states_list = []
                n_slice = (batched_tokens + self.moe_slice_length -
                           1) // self.moe_slice_length
                for i in range(n_slice):
                    s = i * self.moe_slice_length
                    e = batched_tokens if i == (n_slice -
                                    1) else (i + 1) * self.moe_slice_length
                    current_hidden_states = torch.ops.hpu.mixture_of_experts(
                        hidden_states=x_fp8[s:e, ...],
                        expert_routing_table=topk_ids[s:e, ...],
                        router_weights=topk_weights[s:e, ...],
                        w12=self.w13_weight_list,
                        w3=self.w2_weight_list,
                        d_scale_hidden_states=x_scale,
                        d_scale_intermediate_hidden_states=self.w2_input_scale_list,
                        d_scale_w12=self.w13_weight_scale_list,
                        d_scale_w3=self.w2_weight_scale_list,
                        permuted_weights=True,
                        activation="silu",
                        experts_min=ep_shift,
                        experts_max=(num_experts + ep_shift - 1),
                    )
                    final_hidden_states_list.append(current_hidden_states)
                final_hidden_states = torch.cat(final_hidden_states_list, dim=0)
            else:
                final_hidden_states = torch.ops.hpu.mixture_of_experts(
                    hidden_states=x_fp8,
                    expert_routing_table=topk_ids,
                    router_weights=topk_weights,
                    w12=self.w13_weight_list,
                    w3=self.w2_weight_list,
                    d_scale_hidden_states=x_scale,
                    d_scale_intermediate_hidden_states=self.w2_input_scale_list,
                    d_scale_w12=self.w13_weight_scale_list,
                    d_scale_w3=self.w2_weight_scale_list,
                    permuted_weights=True,
                    activation="silu",
                    experts_min=ep_shift,
                    experts_max=(num_experts + ep_shift - 1),
                )
            return final_hidden_states.view(-1, x.shape[1])
        else:
            output = layer.moe_op(
                x,
                topk_ids.to(torch.int64),
                topk_weights.to(x.dtype),
                permuted_weights=True,
                activation=activation,
            )
            return output.view(*input_shape)


fp8.Fp8LinearMethod = Fp8LinearMethod
fp8.Fp8MoEMethod = HPUFp8MoEMethod
