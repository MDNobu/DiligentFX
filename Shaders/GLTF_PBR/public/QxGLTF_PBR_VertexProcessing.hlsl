#ifndef __QxGLTF_PBR_VertexProcessing
#define __QxGLTF_PBR_VertexProcessing

#include "QxGLTF_PBR_Structures.hlsl"

struct QxGLTF_TransformedVertex
{
    float3 WorldPos;
    float3 Normal;
};

float3x3 InverseTranspose3x3(float3x3 M)
{
    // Note that in HLSL, M_t[0] is the first row, while in GLSL, it is the 
    // first column. Luckily, determinant and inverse matrix can be equally 
    // defined through both rows and columns.
    float det = dot(cross(M[0], M[1]), M[2]);
    float3x3 adjugate =
        float3x3(cross(M[1], M[2]),
            cross(M[2], M[0]),
            cross(M[0], M[1]));
    return adjugate / det;
}

QxGLTF_TransformedVertex QxGLTF_TransformVertex(
    in float3 Pos,
    in float3 Normal,
    in float4x4 Transform
    )
{
    QxGLTF_TransformedVertex TransformedVert;

    float4 locPos = mul(Transform, float4(Pos, 1.0));
    float3x3 NormalTransform = (float3x3)(Transform);
    NormalTransform = InverseTranspose3x3(NormalTransform);
    Normal = mul(NormalTransform, Normal);
    TransformedVert.Normal = normalize(Normal);

    //#TODO 这里这个/.w是什么意思
    TransformedVert.WorldPos = locPos.xyz / locPos.w;
    return TransformedVert;
}



#endif