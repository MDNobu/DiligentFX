#include <cstring>
#include <array>

#include "QxGLTF_PBR_Render.hpp"
#include "../../../Utilities/include/DiligentFXShaderSourceStreamFactory.hpp"
#include "CommonlyUsedStates.h"
#include "HashUtils.hpp"
#include "ShaderMacroHelper.hpp"
#include "BasicMath.hpp"
#include "GraphicsUtilities.h"
#include "MapHelper.hpp"
#include "GraphicsAccessories.hpp"

namespace ShaderConstants
{
const char* CameraAttrib = "cbCameraAttribs";

}

namespace Diligent
{
QxGLTF_PBR_Render::QxGLTF_PBR_Render(IRenderDevice* pDevice, IDeviceContext* pCtx, const CreateInfo& CI)
    : m_Settings(CI)
{
    // 准备IBL用的纹理的响应的srv
    if (m_Settings.UseIBL)
    {
        PrecomputeBRDF(pDevice, pCtx);

        TextureDesc TexDesc;
        TexDesc.Name = "Irradiance cube map for GLTF render";
        TexDesc.Type = RESOURCE_DIM_TEX_CUBE;
        TexDesc.Usage = USAGE_DEFAULT;
        TexDesc.BindFlags = BIND_SHADER_RESOURCE | BIND_RENDER_TARGET;
        TexDesc.Width = IrradianceCubeDim;
        TexDesc.Height = IrradianceCubeDim;
        TexDesc.Format = IrradianceCubeFmt;
        TexDesc.ArraySize = 6;
        TexDesc.MipLevels = 0;

        RefCntAutoPtr<ITexture> IrradianceCubeTex;
        pDevice->CreateTexture(TexDesc, nullptr,
            &IrradianceCubeTex);
        m_pIrradianceCubeSRV =
            IrradianceCubeTex->GetDefaultView(
                TEXTURE_VIEW_SHADER_RESOURCE);

        TexDesc.Name = "Prefiltered environment map for GLTF renderer";
        TexDesc.Width = PrefilteredEnvMapDim;
        TexDesc.Height = PrefilteredEnvMapDim;
        TexDesc.Format = PrefilteredEnvMapFmt;
        RefCntAutoPtr<ITexture> PrefilteredEnvMapTex;
        pDevice->CreateTexture(TexDesc, nullptr, &PrefilteredEnvMapTex);
        m_pPrefilteredEnvMapSRV =
            PrefilteredEnvMapTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
    }

    // 创建和初始化black/white/defaultNromalmap/default physic 这几个纹理和响应的SRV/Sampler
    {
        static constexpr Uint32 TexDim = 8;
        TextureDesc TexDesc;
        TexDesc.Name = "White texture for GLTF renderer";
        TexDesc.Type = RESOURCE_DIM_TEX_2D_ARRAY;
        TexDesc.Usage = USAGE_IMMUTABLE;
        TexDesc.BindFlags = BIND_SHADER_RESOURCE;
        TexDesc.Width = TexDim;
        TexDesc.Height = TexDim;
        TexDesc.Format = TEX_FORMAT_RGBA8_UNORM;
        TexDesc.MipLevels = 1;
        std::vector<Uint32> Data(TexDim * TexDim, 0xFFFFFFFF);
        TextureSubResData Level0Data{Data.data(), TexDim * 4};
        TextureData InitData{&Level0Data, 1};
        RefCntAutoPtr<ITexture> pWhiteTex;
        pDevice->CreateTexture(TexDesc, &InitData, &pWhiteTex);
        m_pWhiteTexSRV = pWhiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);

        TexDesc.Name = "Black texture for GLTF renderer";
        for (Uint32& c : Data)
        {
            c = 0;
        }
        RefCntAutoPtr<ITexture> pBlackTex;
        pDevice->CreateTexture(TexDesc, &InitData, &pBlackTex);
        m_pBlackTexSRV = pBlackTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);

        TexDesc.Name = "Default normal map for GLTF renderer";
        for (Uint32& color : Data)
        {
            color = 0x00FF7F7F;
        }
        RefCntAutoPtr<ITexture> pDefaultNormalMap;
        pDevice->CreateTexture(TexDesc, &InitData, &pDefaultNormalMap);
        m_pDefaultNormalMapSRV =
            pDefaultNormalMap->GetDefaultView(
                TEXTURE_VIEW_SHADER_RESOURCE);

        TexDesc.Name = "Default physical descriptioin map for GLTF renderer";
        for (Uint32& color : Data)
        {
            color = 0x0000FF00;
        }
        RefCntAutoPtr<ITexture> pDefaultPhycDescTex;
        pDevice->CreateTexture(TexDesc, &InitData, &pDefaultPhycDescTex);
        m_pDefaultPhysDescSRV =
            pDefaultPhycDescTex->GetDefaultView(
                TEXTURE_VIEW_SHADER_RESOURCE);

        StateTransitionDesc Barriers[] =
        {
            {pWhiteTex, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE,STATE_TRANSITION_FLAG_UPDATE_STATE },
{pBlackTex, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE,STATE_TRANSITION_FLAG_UPDATE_STATE },
{pDefaultNormalMap, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE,STATE_TRANSITION_FLAG_UPDATE_STATE },
{pDefaultPhycDescTex, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE,STATE_TRANSITION_FLAG_UPDATE_STATE },
        };
        pCtx->TransitionResourceStates(_countof(Barriers), Barriers);

        RefCntAutoPtr<ISampler> pDefaultSampler;
        pDevice->CreateSampler(Sam_LinearClamp, &pDefaultSampler);
        m_pWhiteTexSRV->SetSampler(pDefaultSampler);
        m_pBlackTexSRV->SetSampler(pDefaultSampler);
        m_pDefaultNormalMapSRV->SetSampler(pDefaultSampler);
    }

    if (CI.RTVFmt != TEX_FORMAT_UNKNOWN || CI.DSVFmt != TEX_FORMAT_UNKNOWN)
    {
        CreateUniformBuffer(pDevice,
            sizeof(GLTFNodeShaderTransforms),
            "GLTF node transform CB",
            &m_TransformCB);
        CreateUniformBuffer(pDevice,
            sizeof(GLTFMaterialShaderInfo) + sizeof(GLTFRendererShaderParameters),
            "GLTF attribs CB",
            &m_GTLFAttribCB);
        CreateUniformBuffer(
            pDevice,
            static_cast<Uint32>(sizeof(float4x4) * m_Settings.MaxJointCount),
            "GLTF joint transforms",
            &m_JointsBuffer
            );

        StateTransitionDesc Barriers[] =
        {
            {m_TransformCB, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_CONSTANT_BUFFER, STATE_TRANSITION_FLAG_UPDATE_STATE},
{m_GTLFAttribCB, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_CONSTANT_BUFFER, STATE_TRANSITION_FLAG_UPDATE_STATE},
{m_JointsBuffer, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_CONSTANT_BUFFER, STATE_TRANSITION_FLAG_UPDATE_STATE}
        };

        pCtx->TransitionResourceStates(_countof(Barriers), Barriers);
        CreatePSO(pDevice);
    }
}

void QxGLTF_PBR_Render::Render(IDeviceContext* pCtx,
                               GLTF::Model& GLTFModel,
                               const RenderInfo& RenderParams,
                               ModelResourceBindings* pModelBindings,
                               ResourceCacheBindings* pCacheBindings)
{
    DEV_CHECK_ERR((pModelBindings != nullptr) && (pCacheBindings != nullptr),
    "Either model bindings or cache bindings must not be null");
    DEV_CHECK_ERR(pModelBindings == nullptr ||
        pModelBindings->MaterialSRB.size() == GLTFModel.Materials.size(),
        "The number of material shader resource bindings is not consistent with the number of materials");

    // 设置vetex buffer和index buffer
    if (pModelBindings != nullptr)
    {
        std::array<IBuffer*, 2> pVBs =
            {
                GLTFModel.GetBuffer(GLTF::Model::BUFFER_ID_VERTEX_BASIC_ATTRIBS),
                GLTFModel.GetBuffer(GLTF::Model::BUFFER_ID_VERTEX_SKIN_ATTRIBS)
            };
        pCtx->SetVertexBuffers(0,
            static_cast<Uint32>(pVBs.size()),
                pVBs.data(),
                nullptr,
                RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                SET_VERTEX_BUFFERS_FLAG_RESET);

        if (auto* pIndexBuffer = GLTFModel.GetBuffer(GLTF::Model::BUFFER_ID_INDEX))
        {
            pCtx->SetIndexBuffer(
                pIndexBuffer,
                0,
                RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }
    }

    const auto ModelFirstIndexLocation = GLTFModel.GetFirstIndexLocation();
    const auto ModelBaseVertex = GLTFModel.GetBaseVertex();

    const std::array<GLTF::Material::ALPHA_MODE, 3> AlphaModes
    {
        GLTF::Material::ALPHA_MODE_OPAQUE,// Opaque primitives - first
        GLTF::Material::ALPHA_MODE_MASK,// Alpha-masked primitives - second
        GLTF::Material::ALPHA_MODE_BLEND// Transparent primitives - last (TODO: depth sorting)
    };

    const GLTF::Mesh* pLastAnimationMesh = nullptr;
    IPipelineState* pCurPSO = nullptr;
    IShaderResourceBinding* pCurSRB = nullptr;
    PSOKey CurrPSOKey;

    for (GLTF::Material::ALPHA_MODE AlphaMode : AlphaModes)
    {
        for (const GLTF::Node* pNode : GLTFModel.LinearNodes)
        {
            if (!pNode->pMesh)
            {
                continue;
            }

            const GLTF::Mesh& Mesh = *pNode->pMesh;

            // Render mesh primitives
            for (const GLTF::Primitive& primitive : Mesh.Primitives)
            {
                const GLTF::Material& material =
                    GLTFModel.Materials[primitive.MaterialId];
                if (material.Attribs.AlphaMode != AlphaMode)
                {
                    continue;
                }

                // 根据需要更新和设置当前pso,以及清空SRB
                {
                const PSOKey Key{AlphaMode, material.DoubleSided};
                if (Key != CurrPSOKey)
                {
                    CurrPSOKey = Key;
                    pCurPSO = nullptr;
                }
                if (pCurPSO == nullptr)
                {
                    pCurPSO = GetPSO(CurrPSOKey);
                    VERIFY_EXPR(pCurPSO != nullptr);
                    pCtx->SetPipelineState(pCurPSO);
                    pCurSRB = nullptr;
                }
                else
                {
                    VERIFY_EXPR(pCurPSO ==
                        GetPSO(PSOKey{AlphaMode, material.DoubleSided}));
                }
                    
                }


                // 获得draw call要用的SRB并提交
                if (pModelBindings != nullptr)
                {
                    VERIFY(primitive.MaterialId <
                        pModelBindings->MaterialSRB.size(),
                           "Material index is out of bounds. This mostl likely indicates that shader resources were initialized for a different model.");

                    IShaderResourceBinding* const pSRB =
                        pModelBindings->MaterialSRB[primitive.MaterialId]
                            .RawPtr<IShaderResourceBinding>();
                    DEV_CHECK_ERR(pSRB != nullptr, "Unable to find SRB GLTF material");
                    if (pCurSRB != pSRB)
                    {
                        pCurSRB = pSRB;
                        pCtx->CommitShaderResources(
                            pSRB,
                            RESOURCE_STATE_TRANSITION_MODE_VERIFY);
                    }
                }
                else
                {
                    VERIFY_EXPR(pCacheBindings != nullptr);
                    if (pCurSRB != pCacheBindings->pSRB)
                    {
                        pCurSRB = pCacheBindings->pSRB;
                        pCtx->CommitShaderResources(
                            pCurSRB,
                            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
                    }
                }

#pragma region UpdateShaderParameters
                size_t JointCount = Mesh.Transforms.jointMatrices.size();
                if (JointCount > m_Settings.MaxJointCount)
                {
                    LOG_WARNING_MESSAGE("The number of joints in the mesh (", JointCount, ") exceeds the maximum number (", m_Settings.MaxJointCount,
                                    ") reserved in the buffer. Increase MaxJointCount when initializing the renderer.");
                    JointCount = m_Settings.MaxJointCount;
                }

                // 上传node matrix /joint count参数
                {
                    MapHelper<GLTFNodeShaderTransforms> pTransforms{
                        pCtx, m_TransformCB,
                        MAP_WRITE, MAP_FLAG_DISCARD
                        };
                    pTransforms->NodeMatrix =
                        Mesh.Transforms.matrix * RenderParams.ModelTransform;
                    pTransforms->JointCount =
                        static_cast<int>(JointCount);
                }

                // 上传joints matrixes 到gpu
                if (JointCount != 0 && pLastAnimationMesh != &Mesh)
                {
                    MapHelper<float4x4> pJoints{
                        pCtx, m_JointsBuffer,
                        MAP_WRITE, MAP_FLAG_DISCARD
                        };
                    memcpy(pJoints, Mesh.Transforms.jointMatrices.data(),
                        JointCount * sizeof(float4x4));
                    pLastAnimationMesh = &Mesh;
                }

                // 上传材质属性参数和renderer的参数到gpu
                {
                    // 这个用来辅助上传数据到gpu
                    struct GLTFAttribs
                    {
                        GLTFRendererShaderParameters RenderParameters;
                        GLTF::Material::ShaderAttribs MaterialInfo;
                        static_assert(
                            sizeof(GLTFMaterialShaderInfo) == sizeof(GLTF::Material::ShaderAttribs),
                            "The sizeof(GLTFMaterialShaderInfo) is inconsistent with sizeof(GLTF::Material::ShaderAttribs)");
                    };
                    static_assert(
                        sizeof(GLTFAttribs) <= 256,
                        "Size of dynamic GLTFAttribs buffer exceeds 256 bytes. "
                                                              "It may be worth trying to reduce the size or just live with it.");
                    MapHelper<GLTFAttribs> pGLTFAttribs{
                        pCtx, m_GTLFAttribCB,
                        MAP_WRITE, MAP_FLAG_DISCARD
                        };

                    pGLTFAttribs->MaterialInfo = material.Attribs;

                    auto& ShaderParams = pGLTFAttribs->RenderParameters;

                    ShaderParams.DebugViewType =
                        static_cast<int>(m_RenderParams.DebugView);
                    ShaderParams.OcclusionStrength =
                        m_RenderParams.OcclusionStrength;
                    ShaderParams.EmissionScale =
                        m_RenderParams.EmissionScale;
                    ShaderParams.AverageLogLum = m_RenderParams.AverageLogLum;
                    ShaderParams.MiddleGray =
                        m_RenderParams.MiddleGray;
                    ShaderParams.WhitePoint = m_RenderParams.WhitePoint;
                    ShaderParams.IBLScale = m_RenderParams.IBLScale;
                    
                }

#pragma endregion

                // 进行draw call
                if (primitive.HasIndices())
                {
                    DrawIndexedAttribs drawAttrs{
                        primitive.IndexCount,
                        VT_UINT32,
                        DRAW_FLAG_VERIFY_ALL
                        };
                    drawAttrs.FirstIndexLocation =
                        ModelFirstIndexLocation + primitive.FirstIndex;
                    drawAttrs.BaseVertex = ModelBaseVertex;
                    pCtx->DrawIndexed(drawAttrs);
                }
                else
                {
                    DrawAttribs drawAttrs{
                        primitive.VertexCount,
                        DRAW_FLAG_VERIFY_ALL
                        };
                    drawAttrs.StartVertexLocation = ModelBaseVertex;
                    pCtx->Draw(drawAttrs);
                }
            }
        }
    }
}

void QxGLTF_PBR_Render::CreateMaterialSRB(
    GLTF::Model& Model,
    GLTF::Material& Material,
    IBuffer* pCameraAttribs,
    IBuffer* pLightAttribs,
    IPipelineState* pPSO,
    IShaderResourceBinding** ppMaterialSRB)
{
    if (pPSO == nullptr)
    {
        pPSO = GetPSO(PSOKey{});
    }

    pPSO->CreateShaderResourceBinding(ppMaterialSRB, true);
    auto* const pSRB = *ppMaterialSRB;
    if (pSRB == nullptr)
    {
        LOG_ERROR_MESSAGE("Failed to create material SRB");
        return;
    }

    InitCommonSRBVars(pSRB, pCameraAttribs, pLightAttribs);

    auto SetTexture = [&](GLTF::Material::TEXTURE_ID TexId,
        ITextureView* pDefaultTexSRV,
        const char* VarName)
    {
      RefCntAutoPtr<ITextureView> pTexSRV;

        auto TexIdx = Material.TextureIds[TexId];
      if (TexIdx >= 0)
      {
          if (auto* pTexture = Model.GetTexture(TexId))
          {
              if (pTexture->GetDesc().Type == RESOURCE_DIM_TEX_2D_ARRAY)
              {
                pTexSRV = pTexture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);    
              }
              else
              {
                  TextureViewDesc SRVDesc;
                  SRVDesc.ViewType = TEXTURE_VIEW_SHADER_RESOURCE;
                  SRVDesc.TextureDim = RESOURCE_DIM_TEX_2D_ARRAY;
                  pTexture->CreateView(SRVDesc, &pTexSRV);
              }
          }
      }

      if (pTexSRV == nullptr)
      {
          pTexSRV = pDefaultTexSRV;
      }
      if (auto* pVar =
          pSRB->GetVariableByName(SHADER_TYPE_PIXEL, VarName))
      {
          pVar->Set(pTexSRV);
      }
    };

    SetTexture(GLTF::Material::TEXTURE_ID_BASE_COLOR,
        m_pWhiteTexSRV,
        "g_ColorMap");
    SetTexture(GLTF::Material::TEXTURE_ID_PHYSICAL_DESC,
        m_pDefaultPhysDescSRV,
        "g_PhysicalDescriptorMap");
    SetTexture(GLTF::Material::TEXTURE_ID_NORMAL_MAP,
        m_pDefaultNormalMapSRV,
        "g_NormalMap");
    if (m_Settings.UseAO)
    {
        SetTexture(GLTF::Material::TEXTURE_ID_OCCLUSION,
            m_pWhiteTexSRV,
            "g_AOMap");
    }

    if (m_Settings.UseEmissive)
    {
        SetTexture(GLTF::Material::TEXTURE_ID_EMISSIVE,
            m_pBlackTexSRV,
            "g_EmissiveMap");
    }
}

void QxGLTF_PBR_Render::CreatePSO(IRenderDevice* pDevice)
{
    GraphicsPipelineStateCreateInfo PSOCreateInfo;
    PipelineStateDesc& PSODesc =
        PSOCreateInfo.PSODesc;
    GraphicsPipelineDesc& GraphicPipeline =
        PSOCreateInfo.GraphicsPipeline;

    PSODesc.Name = "Render GLTF PBR PSO";
    PSODesc.PipelineType = PIPELINE_TYPE_GRAPHICS;

    GraphicPipeline.NumViewports = 1;
    GraphicPipeline.RTVFormats[0] = m_Settings.RTVFmt;
    GraphicPipeline.DSVFormat = m_Settings.DSVFmt;
    GraphicPipeline.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    GraphicPipeline.RasterizerDesc.CullMode = CULL_MODE_BACK;
    GraphicPipeline.RasterizerDesc.FrontCounterClockwise = m_Settings.FrontCCW;

    ShaderCreateInfo ShaderCI;
    ShaderCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
    ShaderCI.UseCombinedTextureSamplers = true;
    ShaderCI.pShaderSourceStreamFactory = &DiligentFXShaderSourceStreamFactory::GetInstance();

    ShaderMacroHelper Macros;
    Macros.AddShaderMacro("MAX_JOINT_COUNT", m_Settings.MaxJointCount);
    Macros.AddShaderMacro("ALLOW_DEBUG_VIEW", m_Settings.AllowDebugView);
    Macros.AddShaderMacro("TONE_MAPPING_MODE", "TONE_MAPPING_MODE_UNCHARTED2");
    Macros.AddShaderMacro("GLTF_PBR_IBL", m_Settings.UseIBL);
    Macros.AddShaderMacro("GLTF_PBR_USE_AO", m_Settings.UseAO);
    Macros.AddShaderMacro("GLTF_PBR_USE_EMISSIVE", m_Settings.UseEmissive);
    Macros.AddShaderMacro("USE_TEXTURE_ALTAS", m_Settings.UseTextureAtlas);
    Macros.AddShaderMacro("PBR_WORKFLOW_METALLIC_ROUGHNESS", GLTF::Material::PBR_WORKFLOW_METALL_ROUGH);
    Macros.AddShaderMacro("PBR_WORKFLOW_SPECULAR_GLOSINESS", GLTF::Material::PBR_WORKFLOW_SPEC_GLOSS);
    Macros.AddShaderMacro("GLTF_ALPHA_MODE_OPAQUE", GLTF::Material::ALPHA_MODE_OPAQUE);
    Macros.AddShaderMacro("GLTF_ALPHA_MODE_MASK", GLTF::Material::ALPHA_MODE_MASK);
    Macros.AddShaderMacro("GLTF_ALPHA_MODE_BLEND", GLTF::Material::ALPHA_MODE_BLEND);

    ShaderCI.Macros = Macros;
    // create vertex shader
    RefCntAutoPtr<IShader> pVS;
    {
        ShaderCI.Desc.ShaderType = SHADER_TYPE_VERTEX;
        ShaderCI.EntryPoint = "main";
        ShaderCI.Desc.Name = "GLTF PBR VS";
        ShaderCI.FilePath = "RenderGLTF_PBR.vsh";
        pDevice->CreateShader(ShaderCI, &pVS);
    }

    //Create pixel shader
    RefCntAutoPtr<IShader> pPS;
    {
        ShaderCI.Desc.ShaderType = SHADER_TYPE_PIXEL;
        ShaderCI.Desc.Name = "GLTF PBR PS";
        ShaderCI.FilePath = "RenderGLTF_PBR.psh";
        pDevice->CreateShader(ShaderCI, &pPS);
    }

    LayoutElement Inputs[] =
    {
        //float3 Pos     : ATTRIB0;
        {0, 0, 3, VT_FLOAT32},
        //float3 Normal  : ATTRIB1;
        {1, 0, 3, VT_FLOAT32},
        //float2 UV0     : ATTRIB2;
        {2, 0, 2, VT_FLOAT32},
        //float2 UV1     : ATTRIB3;
        {3, 0, 2, VT_FLOAT32},
        //float4 Joint0  : ATTRIB4;
        {4, 1, 4, VT_FLOAT32},
        //float4 Weight0 : ATTRIB5;
        {5, 1, 4, VT_FLOAT32}
    };

    //
    PSOCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = Inputs;
    PSOCreateInfo.GraphicsPipeline.InputLayout.NumElements = _countof(Inputs);

    PSODesc.ResourceLayout.DefaultVariableType =
        SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;

    std::vector<ShaderResourceVariableDesc> Vars =
    {
        {SHADER_TYPE_VERTEX, "cbTransforms", SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
        {SHADER_TYPE_PIXEL, "cbGLTFAttribs", SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
        {SHADER_TYPE_VERTEX, "cbJointTransforms", SHADER_RESOURCE_VARIABLE_TYPE_STATIC}
    };

    std::vector<ImmutableSamplerDesc> ImtblSamplers;
    if (m_Settings.UseImmutableSamplers)
    {
        ImtblSamplers.emplace_back(SHADER_TYPE_PIXEL, "g_ColorMap", m_Settings.ColorMapImmutableSampler);
        ImtblSamplers.emplace_back(SHADER_TYPE_PIXEL, "g_PhysicalDescriptorMap",
            m_Settings.PhysicalMapImmutableSampler);
        ImtblSamplers.emplace_back(SHADER_TYPE_PIXEL, "g_NormalMap",
            m_Settings.NormalMapImmutableSampler);
    }

    if (m_Settings.UseAO)
    {
        ImtblSamplers.emplace_back(SHADER_TYPE_PIXEL, "g_AOMap",
            m_Settings.AOMapImmutableSampler);
    }

    if (m_Settings.UseEmissive)
    {
        ImtblSamplers.emplace_back(SHADER_TYPE_PIXEL, "g_EmissiveMap",
            m_Settings.EmissiveMapImmutableSampler);
    }

    if (m_Settings.UseIBL)
    {
         Vars.emplace_back(SHADER_TYPE_PIXEL, "g_BRDF_LUT",
             SHADER_RESOURCE_VARIABLE_TYPE_STATIC);

        ImtblSamplers.emplace_back(SHADER_TYPE_PIXEL, "g_BRDF_LUT",
            Sam_LinearClamp);
        ImtblSamplers.emplace_back(SHADER_TYPE_PIXEL, "g_IrradianceMap",
            Sam_LinearClamp);
        ImtblSamplers.emplace_back(SHADER_TYPE_PIXEL, "g_PrefilteredEnvMap",
            Sam_LinearClamp);
    }

    PSODesc.ResourceLayout.NumVariables =
        static_cast<Uint32>(Vars.size());
    PSODesc.ResourceLayout.Variables = Vars.data();
    PSODesc.ResourceLayout.NumImmutableSamplers =
        static_cast<Uint32>(ImtblSamplers.size());
    PSODesc.ResourceLayout.ImmutableSamplers =
        ImtblSamplers.empty() ? nullptr : ImtblSamplers.data();

    PSOCreateInfo.pVS = pVS;
    PSOCreateInfo.pPS = pPS;

    // 创建opaque psos
    {
        PSOKey Key{GLTF::Material::ALPHA_MODE_OPAQUE, false};

        RefCntAutoPtr<IPipelineState> pSingleSidedOpaquePSO;
        pDevice->CreateGraphicsPipelineState(PSOCreateInfo,
            &pSingleSidedOpaquePSO);
        AddPSO(Key, std::move(pSingleSidedOpaquePSO));

        PSOCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = CULL_MODE_NONE;

        Key.DoubleSided = true;

        RefCntAutoPtr<IPipelineState> pDoubleSidedOpquePSO;
        pDevice->CreateGraphicsPipelineState(
            PSOCreateInfo,
            &pDoubleSidedOpquePSO
            );

        AddPSO(Key, std::move(pDoubleSidedOpquePSO));
    }

    PSOCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = CULL_MODE_BACK;

    RenderTargetBlendDesc& RT0 = PSOCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0];
    RT0.BlendEnable = true;
    RT0.SrcBlend = BLEND_FACTOR_SRC_ALPHA;
    RT0.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA;
    RT0.BlendOp = BLEND_OPERATION_ADD;
    RT0.SrcBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA;
    RT0.DestBlendAlpha = BLEND_FACTOR_ZERO;
    RT0.BlendOpAlpha = BLEND_OPERATION_ADD;

    {
        PSOKey Key{GLTF::Material::ALPHA_MODE_BLEND, false};

        RefCntAutoPtr<IPipelineState> pSingleSidedBlendPSO;
        pDevice->CreateGraphicsPipelineState(PSOCreateInfo,
            &pSingleSidedBlendPSO);
        AddPSO(Key, std::move(pSingleSidedBlendPSO));

        PSOCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = CULL_MODE_NONE;

        Key.DoubleSided = true;
        RefCntAutoPtr<IPipelineState> pDoubleSidedBlendPSO;
        pDevice->CreateGraphicsPipelineState(PSOCreateInfo,
            &pDoubleSidedBlendPSO);
        AddPSO(Key, std::move(pDoubleSidedBlendPSO));
    }

    // 每个pso cache中的pso 更新shader 参数
    for (RefCntAutoPtr<IPipelineState>& PSO : m_PSOCaches)
    {
        if (m_Settings.UseIBL)
        {
            PSO->GetStaticVariableByName(
                SHADER_TYPE_PIXEL,
                "g_BRDF_LUT")->Set(
                    m_pBRDF_LUT_SRV);
        }

        PSO->GetStaticVariableByName(
            SHADER_TYPE_VERTEX,
            "cbTransforms")->Set(
                m_TransformCB);
        PSO->GetStaticVariableByName(
            SHADER_TYPE_PIXEL,
            "cbGLTFAttribs")->Set(
                m_GTLFAttribCB);
        PSO->GetStaticVariableByName(
            SHADER_TYPE_VERTEX,
            "cbJointTransforms")->Set(
                m_JointsBuffer);
    }
}

void QxGLTF_PBR_Render::InitCommonSRBVars(IShaderResourceBinding* pSRB, IBuffer* pCameraAttribs, IBuffer* pLightAttribs)
{
    VERIFY_EXPR(pSRB != nullptr);

    if (pCameraAttribs != nullptr)
    {
        if (auto* pCameraAttrbsVSVar =
            pSRB->GetVariableByName(SHADER_TYPE_VERTEX, "cbCameraAttribs"))
        {
            pCameraAttrbsVSVar->Set(pCameraAttribs);
        }

        if (auto* pCameraAttribsPSVar =
            pSRB->GetVariableByName(SHADER_TYPE_PIXEL, "cbCameraAttribs"))
        {
            pCameraAttribsPSVar->Set(pCameraAttribs);
        }
    }

    if (pLightAttribs)
    {
        if (auto* pLightAttribsPSVar =
            pSRB->GetVariableByName(SHADER_TYPE_PIXEL, "cbLightAttribs"))
        {
            pLightAttribsPSVar->Set(pLightAttribs);
        }
    }

    if (m_Settings.UseIBL)
    {
        if (auto* pIrradianceMapPSVar =
            pSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_IrradianceMap"))
        {
            pIrradianceMapPSVar->Set(m_pIrradianceCubeSRV);
        }

        if (auto* pPrefilteredEnvMap =
            pSRB->GetVariableByName(SHADER_TYPE_PIXEL,
                "g_PrefilteredEnvMap"))
        {
            pPrefilteredEnvMap->Set(m_pPrefilteredEnvMapSRV);
        }
    }
}
}
