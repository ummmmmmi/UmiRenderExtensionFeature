#ifndef ENDFIELD_TOON_BASE_BUFFER_INCLUDED
#define ENDFIELD_TOON_BASE_BUFFER_INCLUDED

#define SHADOW_SAMPLER sampler_linear_clamp_compare
SAMPLER_CMP(SHADOW_SAMPLER);

float cmp(bool condition)
{
    return condition ? 1.0f : 0.0f;
}

float2 cmp(bool2 condition)
{
    float x = condition.x ? 1.0f : 0.0f;
    float y = condition.y ? 1.0f : 0.0f;
    return float2(x,y);
}

float3 cmp(bool3 condition)
{
    float x = condition.x ? 1.0f : 0.0f;
    float y = condition.y ? 1.0f : 0.0f;
    float z = condition.z ? 1.0f : 0.0f;
    return float3(x,y,z);
}

float normalizeMinMax(float x, float minVal, float maxVal)
{
    float range = max(maxVal - minVal, 0.0001);
    return (x - minVal) * rcp(range);
}

//region old struct
struct DotData
{
    float ndl;
    float ndh;
    float ndv;
    float vdh;
    float ldh;
};

struct VecData
{
    float3 positionOS;
    float3 positionWS;
    float4 positionHCS;
    float3 normalWS_raw;
    float3 tangentWS_raw;
    float3 bitangentWS_raw;

    float3 normalWS;
    float3 tangentWS;
    float3 bitangentWS;

    float3 lightDirWS;
    float3 viewDirWS;
    float3 halfDirWS;
    float3 reflectDirWS;
};

//region new struct
struct Attributes
{
    float4 vertex       :POSITION;
    float3 normal       :NORMAL;
    float4 tangent      :TANGENT;
    float4 color        :COLOR;
    float2 uv0          :TEXCOORD0;
    #if _CHARACTER_VFX_SPECIAL
    float2 uv1          :TEXCOORD1;     //emissionMap uv1
    float2 uv2          :TEXCOORD2;     //smoothNormal
    float2 uv4          :TEXCOORD3;     //index xy
    float2 uv5          :TEXCOORD4;     //index zw
    float2 uv6          :TEXCOORD5;     //weight xy
    float2 uv7          :TEXCOORD6;     //weight zw
    #else
    float2 uv1          :TEXCOORD1;     //smoothNormal
    float2 uv4          :TEXCOORD2;     //index xy
    float2 uv5          :TEXCOORD3;     //index zw
    float2 uv6          :TEXCOORD4;     //weight xy
    float2 uv7          :TEXCOORD5;     //weight zw
    #endif
    UNITY_VERTEX_INPUT_INSTANCE_ID
};

struct Varyings
{
    float4 positionHCS      :SV_POSITION;
    float4 uv               :TEXCOORD0;// xy:uv0 zw:uv1
    float4 normalWS         :TEXCOORD1;
    float4 tangentWS        :TEXCOORD2;
    float4 bitangentWS      :TEXCOORD3;
    float3 positionWS       :TEXCOORD4;
    float4 color            :TEXCOORD5;
    float4 viewDirWS        :TEXCOORD6;
    float3 normalOS         :TEXCOORD7;
    float3 positionOS       :TEXCOORD8;
    // uint   facing           :VFACE;

    // Other Props

    UNITY_VERTEX_INPUT_INSTANCE_ID
    UNITY_VERTEX_OUTPUT_STEREO
};

struct EndFieldLightData
{
    float3 position;    //位置
    float3 direction;   //方向
    float3 color;       //颜色
    float  intensity;   //强度
    float  attenuation; //衰减

    float3 standardColor;          //基础材质灯光颜色
    float  useStandardColor;       //使用 基础材质灯光颜色
    float  useStandardIntenstity;  //使用 基础材质灯光强度
};

struct EndFieldVecData
{
    float3 lightDirWS;
    float3 lightDirWS_XZ;
    float3 viewDirWS;
    float3 cameraForwardWS;
    float3 cameraForwardWS_XZ;
    float3 cameraLeftWS;
    float3 halfDirWS;
    float3 reflectDirWS;
};

struct EndFieldDotData
{
    float NdotL;
    float NdotL_XZ;

    float NdotH;
    float NdotV;
    float VdotH;
    float LdotH;

    float NdotL_clamped;
    float NdotV_clamped;
    float NdotV2_clamped;

    float NgtLdotForwardWS_clamped;
};

struct EndFieldShadowData
{
    float char;
    float scene;
};

struct EndFieldSurface
{
    float2 baseUV;
    float2 baseUV1;
    float3 basecolor;
    float3 basecolor_shadow;
    float alpha;
    float metallic;
    float specularLevel;
    float anisotropy;
    float perceptualsmoothness;
    float perceptualRoughness;
    float perceptualRoughness_rain;//雨水的粗糙度
    float rainMask;//带有雨水区域mask
    float occlusion;
    float3 emission;
    float  depth;

    float anisoValue;
    float anisoValue2;

    float proceduralTangentMask;
    float secondSpecMask;
    float anisotropyFade;

    float hairLine;

    float3 specBitangent;

    float3 normalWS_raw;
    float3 tangentWS_raw;
    float3 bitangentWS_raw;
    float  tangent_W;

    float3x3 TBNWS;

    float3 normalOS;
    float3 normalWS;
    float3 tangentWS;
    float3 bitangentWS;

    float3 normalWS_smooth;

    float3 normalWS_withRain;

    float3 positionOS;
    float3 positionWS;
    float4 positionHCS;

    float2 screenUV;
    float2 screenPos;

    uint renderingLayerMask;
};

struct EndFieldEnvData
{
    float useEnvFeature;
    float envIntensity;
    float3 envDirWS;
    float3 envColor_raw;
    float3 envColor_fixed;
};

struct EndFieldBSDF
{
    float3 diffuse;
    float3 diffuse_shadow;
    float3 diffuse_shadow2;
    float  specularLevel;
    float3 F0;
    float3 F90;
    float  perceptualRoughness;
    float  roughness;
    float  anisotropy;
    float  roughnessT;
    float  roughnessB;
    float  roughness2;
    float  occlusion;

    float4 ramp_NdotL;
    float ramp_NdotV;
};

struct Lighting
{
    float3 diffuse;
    float3 specular;
};

struct AggregateLighting
{
    Lighting direct;
    Lighting indirect;
};

struct LightLoopOutput
{
    float3 diffuseLighting;
    float3 specularLighting;
};

//region BSDF func
void UnpackHairNormalMap(inout real3 normal1, inout real3 normal2,
    real4 packedNormal, real scale1 = 1.0, real scale2 = 1.0
    )
{
    real3 nor1, nor2;
    nor1.xy = packedNormal.xy * 2.0 - 1.0;
    nor1.z = max(1.0e-16, sqrt(1.0 - saturate(dot(nor1.xy, nor1.xy))));
    nor1.xy *= scale1;
    normal1 = nor1;

    nor2.xy = packedNormal.zw * 2.0 - 1.0;
    nor2.z = max(1.0e-16, sqrt(1.0 - saturate(dot(nor2.xy, nor2.xy))));
    nor2.xy *= scale2;
    normal2 = nor2;
}

void ApplyToEndFieldEnvData(float envGlobalIntensity, float4 shAr, float4 shAg, float4 shAb, EndFieldSurface surface,
    inout EndFieldEnvData envData)
{
    //询问"从normal方向看，环境光有多亮？"
    float3 sh01 = max(SHEvalLinearL0L1(surface.normalWS.xyz, shAr, shAg, shAb), 0.0f);

    float3 shColor_raw = envGlobalIntensity * sh01.xyz;

    float3x3 SH01_mtx = float3x3(shAr.xyz, shAg.xyz, shAb.xyz);
    float3 shVector_gray = mul(SH01_mtx, float3(0.2126, 0.7152, 0.0722));
    float3 envLightDirection = SafeNormalize(shVector_gray);
    envLightDirection.y = abs(envLightDirection.y);

    //调整SH颜色
    float3 hsv = RgbToHsv(shColor_raw);

    // 调整饱和度和亮度
    float edgeSmooth = (abs(hsv.x - 0.5f) - 0.45f) * -10.0f;
    float smoothCurve = smoothstep(0, 1, edgeSmooth);

    hsv.y = min(hsv.y, (0.70f - 0.35f * smoothCurve) * saturate(hsv.z));
    hsv.z = 2.0f / (2.0f - hsv.y);

    envData.useEnvFeature = 1.0f;
    envData.envColor_raw = shColor_raw;
    envData.envColor_fixed = HsvToRgb(hsv);
    envData.envDirWS = envLightDirection;

    //询问"从lightDir方向看，环境光有多亮？",得到的就是环境光的最大亮度
    float3 envLightMaxColor = max(SHEvalLinearL0L1(envLightDirection, shAr, shAg, shAb), 0.0f);
    float envLightMaxIntensity = Max3(envLightMaxColor.x, envLightMaxColor.y, envLightMaxColor.z);

    envData.envIntensity = envGlobalIntensity * envLightMaxIntensity;
}

float3 CalculatePureLaserColor(float3 RGB)
{
    float3 hsv = RgbToHsv(RGB);
    float H = hsv.x;  // 色相
    float S = hsv.y;  // 饱和度

    // Step 2: 生成纯彩虹色（饱和度=1，亮度=1的标准彩虹）
    float3 pureRainbow = HsvToRgb(float3(H, 1.0, 1.0));

    // - 基于饱和度在白色(1,1,1)和彩虹色之间插值
    // - 饱和度越高，亮度越强（S=1时放大2倍）
    float brightnessBoost = 2.0 / (2.0 - S);
    float3 boostedColor = brightnessBoost * lerp(1.0, pureRainbow, S);

    return boostedColor;
}

void ApplyEnvFeature(EndFieldSurface surface,
    inout EndFieldEnvData envData)
{
    //_CharacterParams11.w = 1时，不开启雾效
    float envGlobalIntensity = lerp(_EnvironmentGlobalParams0.x, 1, _CharacterParams11.w) * _ExposureParams.x;
    float3 envColor = 0;
    if (_CharacterParams1.y < 0.5)
    {
        float3 lightBoxPos = _IVParam0.xyz;
        float useLightBox = _IVParam0.w;
        float3 lightBoxDir = surface.positionWS.xyz - lightBoxPos;
        float maxLightBoxDistance = Max3(abs(lightBoxDir.z), abs(lightBoxDir.y), abs(lightBoxDir.x));
        //当前存在于lightbox中的系数，0不在，1在中心
        float indirRange = saturate((maxLightBoxDistance - 896.0f) * 0.015625f); //0.015625f = 1 / 64

        float4 shAr, shAg, shAb;

        float cmp2 = cmp(indirRange < 1.0f);
        float cmp3 = 0.0f < useLightBox ? cmp2 : 0;
        //TODO: 以下可能是LPPV，还待优化
        // ===================== tex3 =====================
        if (cmp3 != 0)
        {
            //分两层精度，可能是3dtex 在在同一个lightbox中 有多层精度的
            bool inRange01 = saturate((maxLightBoxDistance - 100.0f) * 0.0833333f) < 1.0f;
            bool inRange02 = saturate((maxLightBoxDistance - 200.0f) * 0.0625f) < 1.0f;
            //0.00390625f = 1 /256, 0.001953125f = 1 / 512, 0.0004882813f = 1 / 2048
            float posSize = inRange01 ? 0.00390625f : (inRange02 ? 0.001953125f : 0.0004882813f);
            float3 T3UV = float3(frac(posSize * surface.positionWS.xyz));
            float4 tex3d_3 = _T3.SampleLevel(sampler_LinearClamp, T3UV, inRange01 ? 0.0f : (inRange02 ? 1.0f : 2.0f));
            float hasLightBox = floor(tex3d_3.w * 255.0f + 0.5f);

            // ===================== tex4、5 =====================
            if (0.0f < hasLightBox)
            {
                //0.00312 = 1 / 320, 0.0125 = 1/ 80
                float3 tex3d_Size = _IVParam1.xyz;
                float3 punctualLightShadowTexV2_UVW = frac(surface.positionWS.xyz / hasLightBox);
                punctualLightShadowTexV2_UVW = punctualLightShadowTexV2_UVW * 4 + 0.5;
                punctualLightShadowTexV2_UVW += floor(mad(tex3d_3.x, 255.0f, 0.5f)) * 5;
                punctualLightShadowTexV2_UVW *= tex3d_Size;

                float4 pLightShadow00 = _T4.SampleLevel(sampler_LinearClamp, punctualLightShadowTexV2_UVW, 0.0f);

                float4 pLightShadow01 = _T5.SampleLevel(sampler_LinearClamp, float3(punctualLightShadowTexV2_UVW.xy, punctualLightShadowTexV2_UVW.z * 0.3333333f), 0.0f);
                float4 pLightShadow02 = _T5.SampleLevel(sampler_LinearClamp, float3(punctualLightShadowTexV2_UVW.xy, punctualLightShadowTexV2_UVW.z * 0.3333333f + 0.3333333f), 0.0f);
                float4 pLightShadow03 = _T5.SampleLevel(sampler_LinearClamp, float3(punctualLightShadowTexV2_UVW.xy, punctualLightShadowTexV2_UVW.z * 0.3333333f + 0.6666667f), 0.0f);

                float3 _81_890 = pLightShadow00.x * (pLightShadow01.xyz * 4.0f - 2.0f);
                float3 _82_567 = pLightShadow00.y * (pLightShadow02.xyz * 4.0f - 2.0f);
                float3 _83_123 = pLightShadow00.z * (pLightShadow03.xyz * 4.0f - 2.0f);

                float4 ar = float4(_81_890, pLightShadow00.x);
                float4 ag = float4(_82_567, pLightShadow00.y);
                float4 ab = float4(_83_123, pLightShadow00.z);

                shAr = lerp(ar, _IVDefaultSHAr, indirRange);
                shAg = lerp(ag, _IVDefaultSHAg, indirRange);
                shAb = lerp(ab, _IVDefaultSHAb, indirRange);
            }
            else
            {
                shAr = _IVDefaultSHAr;
                shAg = _IVDefaultSHAg;
                shAb = _IVDefaultSHAb;
            }
        }
        else
        {
            shAr = _IVDefaultSHAr;
            shAg = _IVDefaultSHAg;
            shAb = _IVDefaultSHAb;
        }

        shAr = _AmbientProbeData[0];   //test code
        shAg = _AmbientProbeData[1];   //test code
        shAb = _AmbientProbeData[2];   //test code

        ApplyToEndFieldEnvData(envGlobalIntensity, shAr, shAg, shAb, surface, envData);
    }
    else
    {
        // ===================== 属于是不使用sh color来附加一下变化的保底做法（副分支：cb0[161].y ≥ 0.5） =====================
        //采样t15 matcap做法的镭射
        if (1.5f < _CharacterParams1.y)
        {
            float3 laserNormalVS = TransformWorldToViewNormal(surface.normalWS, true);
            float4 laserMap = SAMPLE_TEXTURE2D(_LaserMap, sampler_LaserMap, laserNormalVS.xy * 0.5 + 0.5);

            float3 laser = CalculatePureLaserColor(laserMap.xyz);
            envColor = lerp(1.0f, laser, _CharacterParams2.w);    //强度控制
        }
        else
        {
            float3 standardEnvColor = _CharacterParams2.xyz;
            envColor.xyz = standardEnvColor;     //_CharacterParams2
        }

        envData.envColor_fixed = envColor;
        envData.envColor_raw = 1.0f;
        envData.envDirWS = 0.0f;
        envData.useEnvFeature = 0.0f;
        envData.envIntensity = envGlobalIntensity;
    }
}

float4 SampleRampColor_NdotL(EndFieldVecData vecData, EndFieldDotData dotData)
{
    // 视角修正 (View Dependent Bias)
    float viewPitchBias = smoothstep(0, 1, (0.75f - abs(vecData.cameraForwardWS.y)) * 2.0f);   //视角保持水平为1，其余角度为0
    float LdotCamForDir = dotData.NgtLdotForwardWS_clamped;   //负光照方向 dot 视线方向，等于在看向光源位置是1，其余是0

    float backFactor = viewPitchBias * LdotCamForDir;
    float NdotL_Bias = dotData.NdotL * (-dotData.NdotL * 0.5) + 0.5f;

    // 给NdotL的背面（负数区域）加上光照效果, 视角在光照背面时，将明暗交界线推移到后方，有更多细节
    float NdotL_Fixed = dotData.NdotL + lerp(backFactor * NdotL_Bias, _CharacterParams4.w, _CharacterParams3.w);    //启用的开关，是否有更多细节，是否简化细节
    NdotL_Fixed = clamp(NdotL_Fixed, -1, 1);

    float2 rampUV;
    rampUV.x = NdotL_Fixed * 0.5 + 0.5;
    rampUV.y = 0.5;
    float4 rampColor = SAMPLE_TEXTURE2D_LOD(_ShadowRampTex, sampler_ShadowRampTex, rampUV, 0);      //控制颜色变化的ramp

    return rampColor;
}

float SampleRampColor_NdotV(EndFieldVecData vecData, EndFieldSurface surface)
{
    float3 cameraForwardWS_fixed = float3(vecData.cameraForwardWS.x,
                                      min(vecData.cameraForwardWS.y + 0.25f, 1.0f),
                                          vecData.cameraForwardWS.z);
    float3 cameraOffsetDir_nlz = normalize(cameraForwardWS_fixed);
    float2 rampUV_alpha;
    rampUV_alpha.x = dot(surface.normalWS, cameraOffsetDir_nlz) * 0.5 + 0.5;    //NdotV的微调版
    rampUV_alpha.y = 1.0f;
    float rampAlpha = SAMPLE_TEXTURE2D_LOD(_ShadowRampTex, sampler_ShadowRampTex, rampUV_alpha, 0).a;   //控制亮暗渐变程度的ramp
    return rampAlpha;
}

void CalculateSpecularBitangent(EndFieldVecData vecData, inout EndFieldSurface surface)
{
    //模拟围绕 Y 轴的切线流
    float3 worldAxisX = normalize(UNITY_MATRIX_M._m00_m10_m20); // 模型 Right
    float3 worldAxisY = normalize(UNITY_MATRIX_M._m01_m11_m21); // 模型 Up
    float3 worldAxisZ = normalize(UNITY_MATRIX_M._m02_m12_m22); // 模型 Forward

    float3 anisoModelvec = _AnisotropyDirX * worldAxisX + worldAxisY;
    // float3 anisoModelvec = _AnisotropyDirX * float3(UNITY_MATRIX_M[0].zxy) + float3(UNITY_MATRIX_M[1].zxy);
    // float3 anisoModelvec = _AnisotropyDirX * float3(UNITY_MATRIX_I_M[0].xyz) + float3(UNITY_MATRIX_I_M[1].xyz);
    // float3 anisoModelvec = _AnisotropyDirX * float3(UNITY_MATRIX_I_M[0].xyz) + float3(UNITY_MATRIX_I_M[1].xyz);
    float3 anisoModelvec_nlz = SafeNormalize(anisoModelvec);

    float3 specTangent = cross(surface.normalWS_smooth, anisoModelvec_nlz);
    specTangent = lerp(specTangent, surface.tangentWS_raw.yzx, surface.proceduralTangentMask);
    // return float4(dot(specTangent, vecData.viewDirWS).xxx, 1);

    float specTangent_w = lerp(1.0f, surface.tangent_W, surface.proceduralTangentMask);
    surface.specBitangent = cross(surface.normalWS_smooth, specTangent) * specTangent_w * GetOddNegativeScale();

    // 获取模型空间的轴 (用于投影)
    float3 objAxisX_WS = worldAxisX;
    float3 objAxisZ_WS = worldAxisZ;

    // 将法线投影到模型空间 XZ 平面
    float N_dot_ObjX = dot(surface.normalWS_smooth, objAxisX_WS);
    float N_dot_ObjZ = dot(surface.normalWS_smooth, objAxisZ_WS);

    // 将视线投影到模型空间 XZ 平面
    float V_dot_ObjX = dot(vecData.viewDirWS, objAxisX_WS);
    float V_dot_ObjZ = dot(vecData.viewDirWS, objAxisZ_WS);

    // 归一化投影向量 (相当于 normalize(N.xz) 和 normalize(V.xz))
    float invLenN = rsqrt(dot(float2(N_dot_ObjX, N_dot_ObjZ), float2(N_dot_ObjX, N_dot_ObjZ)));
    float invLenV = rsqrt(dot(float2(V_dot_ObjX, V_dot_ObjZ), float2(V_dot_ObjX, V_dot_ObjZ)));

    // 计算两个投影向量的点积 (Cos Angle)
    float planarCos = dot(
        float2(invLenN * N_dot_ObjX, invLenN * N_dot_ObjZ),
        float2(invLenV * V_dot_ObjX, invLenV * V_dot_ObjZ)
    );
    // 计算衰减系数: _AnisotropyEdgeFade 控制边缘淡出强度
    surface.anisotropyFade = pow(saturate(planarCos), _AnisotropyEdgeFade);
}

float AnisotropicSpecular(float3 T, float3 H)
{
    float dotTH = dot(T, H);
    float sinTH = sqrt(1.0f - dotTH * dotTH);
    return max(sinTH, 9.9999997473787516355514526367188e-05f);;
}

float3 Hair_D_EndField(EndFieldVecData vecData, EndFieldSurface surface)
{
    float3 anisoDirWS = ShiftTangent(surface.specBitangent, surface.normalWS_smooth, surface.anisoValue);
    float anisoSpec = AnisotropicSpecular(anisoDirWS, vecData.halfDirWS);
    float HdotAniso = dot(anisoDirWS, vecData.halfDirWS);
    float powerAnisoRange1 = 1.0f * 200.0f;
    float specTerm1 = saturate(surface.specularLevel * pow(anisoSpec, powerAnisoRange1));

    // 应用 Fade (极点衰减)
    float fadeFactorSquared = surface.anisotropyFade * surface.anisotropyFade;

    float2 specRampUV;
    specRampUV.x = specTerm1;
    specRampUV.y = (0.0f < HdotAniso) ? fadeFactorSquared : 0.0f;
    float3 specRampColor = SAMPLE_TEXTURE2D_LOD(_SpecRampMap, sampler_SpecRampMap, specRampUV, 0).xyz;
    float3 specColor_small = surface.anisotropyFade * specTerm1 * specRampColor;
    return specColor_small;
}

float3 Hair_F_EndField(EndFieldVecData vecData, EndFieldSurface surface)
{
    float3 anisoDirWS2 = ShiftTangent(surface.specBitangent, surface.normalWS_smooth, surface.anisoValue2);
    float anisoSpec2 = AnisotropicSpecular(anisoDirWS2.xyz, vecData.halfDirWS);

    float powerAnisoRange2 = trunc(max(1.0f  - _AnisotropyRange2, 0.0f) * 200.0f);
    float specTerm2 = surface.anisotropyFade * pow(anisoSpec2, powerAnisoRange2);
    float3 specColor_large = specTerm2 * surface.secondSpecMask * _AnisotropyColor2.xyz;
    return specColor_large;
}

#define _cb1_13 float4(0.00, 0.00, 0.00, 0.00)   //rainFeature的未知数据

//应用下雨淋湿材质的feature
void ApplyRainFeature(inout EndFieldSurface surface)
{
    float rainStrengthCombined = lerp(_cb1_13.x + _RainEffectIntensity,       _CharacterParams10.y, _CharacterParams10.x);    //主控制强度的变化
    float wetHeightCombined    = lerp(_cb1_13.z + _WetEffectWorldSpaceHeight, _CharacterParams10.w, _CharacterParams10.x);    //可能是水面深度的控制
    float wetIntensityCombined = lerp(_cb1_13.y + _WetEffectIntensity,                           1, _CharacterParams10.x);    //不清楚

    // 2. 计算高度遮罩 (Height Mask)
    float heightMask = smoothstep(0, 1, 2.85714269 * (wetHeightCombined - surface.positionWS.y + 0.2f));

    float wetnessFactor = heightMask * wetIntensityCombined;                  //这两个是一对，1163是另一个模块
    float totalRainFactor = heightMask * wetIntensityCombined + rainStrengthCombined ;  //其中heightMask * wetIntensityCombined模块是控制水深度的

    if (9.99999975e-005 < totalRainFactor)
    {
        float rainMaxParam   = max(wetnessFactor, rainStrengthCombined);
        float wetAlbedoMix   = 1 - rainMaxParam * 0.2;

        // 应用变暗
        surface.basecolor = surface.basecolor * wetAlbedoMix;//亮面albedo，有雨水变色、淋湿变色。淋湿区域，深颜色会变浅，浅颜色会变深
        surface.basecolor_shadow = surface.basecolor_shadow * wetAlbedoMix;   //阴影albedo只受整体湿润影响，因为在暗面不受雨水效果

        surface.rainMask = rainMaxParam + 1.0f;//受粗糙度、金属度、开关控制的强度
    }
}

//用用特殊效果feature
float3 ApplyVFXFeature(EndFieldVecData vecData, EndFieldSurface surface)
{
    #ifdef _CHARACTER_VFX_SPECIAL
        // 计算基础透明度参考值
        float globalAlphaRef = _VFXColor.w * _VFXColorAlpha;
        float2 MapUV = _VFXMainUVSet > 0.5f ? surface.baseUV1 : surface.baseUV;

        float2 VFXMainUV = MapUV + _VFXParams0.w * _VFXSpecialParam.zw;
        VFXMainUV = TRANSFORM_TEX(VFXMainUV, _VFXSpecialMainTex);
        float4 VFXMainMap = SAMPLE_TEXTURE2D(_VFXSpecialMainTex, sampler_VFXSpecialMainTex, VFXMainUV);

        float2 VFXBlendUV = VFXMainMap.xx + MapUV;
        VFXBlendUV = VFXBlendUV + _VFXParams0.w * _VFXSpecialParam.xy;
        VFXBlendUV = TRANSFORM_TEX(VFXBlendUV, _VFXSpecialBlendTex);
        float4 VFXBlendMap = SAMPLE_TEXTURE2D(_VFXSpecialBlendTex, sampler_VFXSpecialBlendTex, VFXBlendUV);

        // 通道选择: 根据 .x 参数在 Flow图的 A通道 和 R通道 之间插值
        float flowChannel = lerp(VFXBlendMap.w, VFXBlendMap.x, _UseVFXMainTexAsAlpha);
        // 计算不透明度遮罩 (Opacity Mask)
        // 逻辑: (全局Alpha * 流动值 + 主图Alpha) * 混合Tint的Alpha
        float opacityMask = saturate((globalAlphaRef * flowChannel + VFXMainMap.w) * _VFXBlendTint.w);

        // 计算混合图颜色基底
        float3 flowBaseColor = lerp(1.0f, VFXBlendMap.xyz, _UseVFXMainTexAsAlpha);
        // 合成基础火焰颜色 (baseFireColor Base Color)
        // Part A: 主图RGB * 不透明度 * TintColor
        // Part B: 流动图RGB * VFX主色 * 强度乘数(.z)
        float3 baseFireColor = VFXMainMap.xyz * opacityMask * _VFXBlendTint.xyz
                            + flowBaseColor.xyz * _VFXColor.xyz * _VFXColorIntensity;

        // 计算溶解阈值 (Mapping 0..1 to -1.01..1.02)
        float dissolveThreshold = _SpecialDissolveScheduleOffset * 2.02f - 1.01f;
        // 计算溶解边缘强度 (Edge Factor)
        // 注意: 这里计算的是 "阈值 - 噪声值"。
        float dissolveEdgeFactor = saturate(dissolveThreshold - VFXMainMap.x);

        // 叠加边缘发光色
        // 如果处于溶解边缘内部，叠加 _VFXFresnelColor 作为高亮
        float3 fireColorWithEdge = lerp(baseFireColor, dissolveEdgeFactor * _VFXFresnelColor.xyz + _VFXColorIntensity, dissolveEdgeFactor);

        // 计算视线与法线的点积 (Safe NdotV)
        float VdotN_safe = saturate(dot(vecData.viewDirWS, surface.normalWS_raw/* 这里原本是normalWS_raw_safe ，原始法线的SafeNormalize版本*/) + _VFXFresnelBias);
        VdotN_safe = pow(VdotN_safe, _VFXFresnelPower);

        // 菲涅尔反转控制 (Flip)
        float vfxFresnel = lerp(1 - VdotN_safe, VdotN_safe, _VFXFresnelFlip);
        // 计算最终流动遮罩 (Flow Mask)
        float visibilityMask = lerp(1.0f, vfxFresnel, _VFXFresnelAffectOpacity);    //菲涅尔是否影响透明度
        float dissolveCutout = saturate(VFXMainMap.x - dissolveThreshold);
        float flowMask = visibilityMask * saturate(flowChannel * globalAlphaRef * dissolveCutout);

        float3 vfxFinalColor = lerp(fireColorWithEdge, _VFXFresnelColor.xyz, vfxFresnel * _VFXFresnelColor.w) * flowMask;  //r10.xyz 熔岩颜色
        float3 effectColor = surface.emission + vfxFinalColor;
    #else
        float3 effectColor = surface.emission;
    #endif
    return effectColor;
}

//手动放置rim灯光位置的feature
float3 ApplyRimFeature(float3 diffuseSat, EndFieldVecData vecData, EndFieldShadowData shadowData, EndFieldSurface surface)
{
    float3 rimLightPosWS = UNITY_MATRIX_M._m03_m13_m23; //_RimLightPosWS
    float3 customDirWS;
    customDirWS.xz = surface.positionWS.xz - rimLightPosWS.xz;
    customDirWS.y = 0;
    customDirWS = SafeNormalize(customDirWS);

    bool enable_Rim = cmp(0.00999999978 < _CharacterParams8.w);
    //rim
    float3 normalVS = mul(UNITY_MATRIX_V, surface.normalWS);
    float3 normalVS_nlz = 0;
    normalVS_nlz.xy = normalize(normalVS.xy);

    //cb0_v67 2560.00, 1440.00, 1.0003906, 1.0006944 1072 float4
    float4 scaledScreenParams = GetScaledScreenParams(); // CB0_m0[67u].xyzw
    float4 screenParams = _ScreenParams; // CB0_m0[62u].xyzw
    float aspectRatio = scaledScreenParams.y / scaledScreenParams.x;

    float2 offsetUV = float2(
        normalVS_nlz.x * aspectRatio * _CharacterParams7.w * 0.006f,
        normalVS_nlz.y * 1.0f        * _CharacterParams7.w * 0.006f
        );

    // cb0_v62 2560.00, 1440.00, 0.0003906, 0.0006944 992 float4
    // float2 currentScreenUV = i.positionHCS.xy * screenParams.zw;
    float2 currentScreenUV = surface.screenUV;
    float2 sampleUV = currentScreenUV + offsetUV;

    float2 clampedSampleUV = clamp(sampleUV.xy, scaledScreenParams.zw - 1.0f, 2.0f - scaledScreenParams.zw);


    float rawDepthSample = SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, sampler_LinearClamp, clampedSampleUV, 0.0f).x;

    float linearDepthSample = LinearEyeDepth(rawDepthSample, _ZBufferParams);
    float linearDepthCurrent = surface.positionHCS.w;
    float rimRange = (linearDepthSample - linearDepthCurrent - 0.1f) * 10.0f;
    float rimRange_smt = smoothstep(0, 1, rimRange);

    //dot distance
    float rim_atten = saturate(1 + dot(customDirWS.xz, vecData.cameraLeftWS.xz));
    rim_atten = Min3(rim_atten, surface.occlusion, shadowData.char);

    float NdotRimDir_clamp = saturate(dot(vecData.cameraLeftWS, surface.normalWS));

    //mix rim color
    float3 rimDiffuse = lerp(0.25, diffuseSat, _CharacterParams6.w) * NdotRimDir_clamp;
    float3 rimColor = rim_atten * rimRange_smt * _CharacterParams8.xyz * _CharacterParams8.w;      //_cb0_168.xyz: rim color, _cb0_168.w:rim intensity
    float3 rimFinalColor = rimDiffuse * rimColor;
    rimFinalColor = enable_Rim ? rimFinalColor : 0;
    return rimFinalColor;
}

//根据环境球谐自动添加rim的feature
float3 ApplySHRimFeature(float3 diffuseSat, float3 saturatedDirLightColor, EndFieldVecData vecData, EndFieldShadowData shadowData, EndFieldDotData dotData,
    EndFieldEnvData envData, EndFieldSurface surface, EndFieldBSDF bsdf)
{
    float SHdotN = dot(envData.envDirWS, surface.normalWS) * envData.useEnvFeature;

    // 视线与法线夹角的边缘蒙版 (Fresnel-like Edge Mask)
    float VdotN = -abs(dot(vecData.viewDirWS, surface.normalWS));
    float edgeFresnelMask = smoothstep(0, 1, 5.0f * (VdotN + 0.40f));

    // 背光强度因子 (Backlight Factor)
    // 在场景阴影中(shadowData.scene=0)倾向于使用1
    // 在亮部(shadowData.scene=1)倾向于使用负灯光与视角的dot，也就是看向主光方向（逆光）时才是1
    float backlightFactor = lerp(1, dotData.NgtLdotForwardWS_clamped, shadowData.scene);

    // 法线与光照方向的匹配蒙版 (Normal-Light Alignment)
    // 在场景阴影中(shadowData.scene=0)倾向于使用环境光方向的dot
    // 在亮部(shadowData.scene=1)倾向于使用主光的dot
    float lightRangeDot = 0.5 - dotData.NdotL_XZ * (dotData.NdotL_XZ * 0.5 - 1);    //拟合曲线
    float rimLightDirectionMask = saturate(lerp(SHdotN, lightRangeDot, shadowData.scene));
    // return float4(SHdotN.xxx, 1);

    // 固有色亮度蒙版 (Albedo Brightness Mask)
    // 在场景阴影中(shadowData.scene=0)倾向于使用1
    // 在亮部(shadowData.scene=1)倾向于使用钳制的颜色权重曲线， 防止边缘光出现在本来就很黑的材质上（这会让黑色看起来发灰）
    float albedoBrightnessMask = smoothstep(0, 1, -16.666666 * (Luminance(bsdf.diffuse) - 0.1f));
    albedoBrightnessMask = lerp(1, albedoBrightnessMask, shadowData.scene);

    // [Color] 边缘光颜色源 (Rim Source Color)
    // 在场景阴影中(shadowData.scene=0)倾向于使用环境光(SHavgColor)
    // 在亮部(shadowData.scene=1)倾向于使用主光(directionalLightColor)
    float SHavgValue = max(Max3(envData.envColor_raw.x, envData.envColor_raw.y, envData.envColor_raw.z) * 0.5f, 1.0f);
    float3 SHavgColor = envData.envColor_raw * rcp(SHavgValue);
    float3 rimSourceColor = lerp(SHavgColor, saturatedDirLightColor, shadowData.scene);

    float shRimCtrlWeight = 1.0f - _CharacterParams3.w;
    // 综合边缘光强度 (Final Rim Intensity)
    float SHRimIntensity = albedoBrightnessMask * min(shadowData.char, surface.occlusion)
                             * edgeFresnelMask * rimLightDirectionMask * backlightFactor
                             * shRimCtrlWeight;

    return SHRimIntensity * rimSourceColor * max(0.15, diffuseSat);
}
//endregion

//region 初始化数据
void InitializeEndFieldLightData(Light light,
    inout EndFieldLightData lightData)
{
    lightData.useStandardColor = _CharacterParams5.w;
    lightData.standardColor = lerp(light.color, _CharacterParams5.xyz, lightData.useStandardColor);
    lightData.useStandardIntenstity = _CharacterParams11.w;
    lightData.direction = light.direction;
    lightData.attenuation = light.distanceAttenuation;  //EndField里是1.6243868
    lightData.intensity = lerp(lightData.attenuation, 1.0f, lightData.useStandardIntenstity);
    lightData.color = lightData.standardColor * lightData.intensity;
}

void IntializeEndFieldShadow(EndFieldSurface surface,
    inout EndFieldShadowData shadowData)
{
    float shadowAttenuation_scene = 1;
    float shadowAttenuation_char = 1;

    // 角色的屏幕空间自阴影
    #ifdef _RAYTRACING_SHADOWS
    shadowAttenuation_scene = SAMPLE_TEXTURE2D(_ScreenSpaceShadowmapTexture, sampler_PointClamp, surface.screenUV).x;
    #else
    shadowAttenuation_scene = SAMPLE_TEXTURE2D(_ScreenSpaceShadowmapTexture, sampler_PointClamp, surface.screenUV).x;
    #ifdef _PEROBJECT_SCREEN_SPACE_SHADOW
    shadowAttenuation_char = SamplePerObjectScreenSpaceShadowmap(surface.screenUV);
    // shadowAttenuation_char = min(shadowAttenuation_scene, SamplePerObjectScreenSpaceShadowmap(surface.screenUV));
    #endif
    #endif

    float enableSceneShadow = _DirectionalShadowParams2.x;
    float defaultSceneShadow = _DirectionalShadowParams2.z;
    float sceneShadowIntensity = _DirectionalShadowParams.x;
    float sceneShadowIntensity_volume = _CharacterParams1.z;

    // 场景的投影
    // shadowAttenuation_scene = sampleTex.x;
    float shadowMaskVal = 0.0f < enableSceneShadow ? shadowAttenuation_scene : defaultSceneShadow;
    float shadowMaskMixed = lerp(1, shadowMaskVal, sceneShadowIntensity);
    float sceneShadowAtten = lerp(shadowMaskMixed, 1, sceneShadowIntensity_volume);
    shadowData.char = shadowAttenuation_char;
    shadowData.scene = sceneShadowAtten;
    shadowData.scene = shadowAttenuation_scene;     //test code
    // shadowData.scene = step(0.5, surface.screenUV.y);    //test code
}

void InitializeEndFieldVecData(Varyings i, EndFieldLightData lightData, EndFieldShadowData shadowData, EndFieldSurface surface,
    inout EndFieldVecData vecData)
{
    float  useCustomLight = _CharacterParams1.w;
    float3 customLightDir = _CharacterParams4.xyz;
    vecData.lightDirWS = lerp(normalize(lightData.direction), customLightDir, useCustomLight);
    vecData.lightDirWS_XZ = SafeNormalize(float3(vecData.lightDirWS.x, 0, vecData.lightDirWS.z));
    vecData.viewDirWS = SafeNormalize(i.viewDirWS.xyz);
    vecData.cameraForwardWS = UNITY_MATRIX_I_V._m02_m12_m22;//UNITY_MATRIX_I_V(会同步scene的矩阵)，但_InvViewMatrix只与游戏相机相关
    vecData.cameraForwardWS_XZ = normalize(float3(vecData.cameraForwardWS.x, 0, vecData.cameraForwardWS.z));
    float3 rimAxis = _CharacterParams9.xyz;
    vecData.cameraLeftWS = SafeNormalize(cross(vecData.cameraForwardWS, rimAxis));

    float3 shiftedLightDirOffset;
    shiftedLightDirOffset.x = vecData.cameraForwardWS.x;
    shiftedLightDirOffset.y = lerp(0.5f, vecData.lightDirWS.y, shadowData.scene);
    shiftedLightDirOffset.z = vecData.cameraForwardWS.z;
    float3x3 matrix_m = float3x3(UNITY_MATRIX_I_M[0].xyz, UNITY_MATRIX_I_M[1].xyz, UNITY_MATRIX_I_M[2].xyz);
    shiftedLightDirOffset = mul(matrix_m, shiftedLightDirOffset);
    float3 lightDirFinal = vecData.lightDirWS * shadowData.scene + shiftedLightDirOffset * 2.0f;//果然是使用了跟随相机前向的跟踪高光方向
    float3 halfDirWS = SafeNormalize(lightDirFinal) + vecData.viewDirWS;
    vecData.halfDirWS = SafeNormalize(halfDirWS);

    vecData.reflectDirWS = reflect(-vecData.viewDirWS, surface.normalWS_withRain);
}

void IntializeEndFieldSurface(Varyings i, uint facing,
    inout EndFieldSurface surface)
{
    surface.depth = i.positionHCS.z;
    surface.baseUV = i.uv.xy;
    surface.baseUV1 = i.uv.zw;
    surface.positionOS = i.positionOS;
    surface.positionWS = i.positionWS;
    surface.positionHCS = i.positionHCS;
    surface.screenUV = GetNormalizedScreenSpaceUV(i.positionHCS.xy);
    surface.screenPos = (uint2)i.positionHCS.xy;

    float4 mainTex = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, surface.baseUV);
    float4 pbrMask = SAMPLE_TEXTURE2D(_PBRMask, sampler_PBRMask, surface.baseUV);
    // float3 bumpTS = UnpackNormalScale(SAMPLE_TEXTURE2D(_NormalMap, sampler_NormalMap, surface.baseUV), _NormalScale);
    float4 bumpMap = SAMPLE_TEXTURE2D(_SplitNormalMap, sampler_SplitNormalMap, surface.baseUV);
    float3 bumpTS = 0;
    float3 bumpTS_smooth = 0;
    UnpackHairNormalMap(bumpTS, bumpTS_smooth,
        bumpMap, _NormalScale, _SpecBumpScale);
    float4 lineMap = SAMPLE_TEXTURE2D(_LineMap, sampler_LineMap, TRANSFORM_TEX(surface.baseUV, _LineMap));
    // float3 emissionTex = SAMPLE_TEXTURE2D(_EmissionMap, sampler_EmissionMap, surface.baseUV).xyz;

    //params part
    surface.basecolor = mainTex.rgb * _BaseColor.rgb;
    surface.alpha     = mainTex.a * _BaseColor.a;
    surface.basecolor_shadow = surface.basecolor * _ShadowColorBrightness;
    surface.basecolor_shadow = lerp(Luminance(surface.basecolor_shadow), surface.basecolor_shadow, _ShadowColorSaturation);
    surface.metallic = 0.0f;// lerp(0, 1, _Metallic);
    surface.specularLevel = pbrMask.g;
    // surface.anisotropy = 0;
    surface.perceptualsmoothness = 0;
    surface.perceptualRoughness = lerp(0, _Roughness, 1 - surface.perceptualsmoothness);
    // surface.perceptualRoughness_rain = 0.01f;
    surface.rainMask = 1.0f;
    surface.occlusion = pbrMask.b;
    // surface.emission         = emissionTex * _EmissionCol.xyz * _EmissionIntensity;

    surface.anisoValue = _AnisotropyValue * 2.0f - 1.0f;
    surface.anisoValue2 = _AnisotropyValue2 * 2.0f - 1.0f;

    surface.proceduralTangentMask = pbrMask.x;
    surface.secondSpecMask = pbrMask.w;
    surface.anisotropyFade = 0;

    surface.hairLine = lineMap.x;

    surface.specBitangent = 0;

    //vector part
    float normalLength = length(i.normalWS.xyz);
    float invNormalLength = 1.0 / max(FLT_MIN, normalLength);

    float tangentWS_w = i.tangentWS.w > 0 ? 1 : -1;
    tangentWS_w *= GetOddNegativeScale();

    float faced = facing ? 1 : _BackFaceNormalFlip * 2 - 1;

    surface.normalWS_raw         = i.normalWS.xyz * rcp(normalLength) * faced;
    surface.tangentWS_raw        = i.tangentWS.xyz * invNormalLength;
    surface.bitangentWS_raw      = cross(i.normalWS.xyz, i.tangentWS.xyz) * tangentWS_w * invNormalLength;
    surface.tangent_W            = i.tangentWS.w;
    surface.TBNWS                = float3x3(surface.tangentWS_raw,
                                            surface.bitangentWS_raw,
                                            i.normalWS.xyz * invNormalLength);
    surface.normalOS             = i.normalOS;
    surface.normalWS             = normalize(TransformTangentToWorld(bumpTS, surface.TBNWS)) * faced;
    surface.tangentWS            = 0;
    surface.bitangentWS          = 0;

    float3x3 TBNWS               = float3x3(surface.tangentWS_raw,
                                            surface.bitangentWS_raw,
                                            surface.normalWS_raw);
    surface.normalWS_smooth             = normalize(TransformTangentToWorld(bumpTS_smooth, TBNWS));

    surface.normalWS_withRain    = surface.normalWS;
}

void InitializeEndFieldDotData(EndFieldVecData vecData, EndFieldSurface surface,
    inout EndFieldDotData dotData)
{
    dotData.NdotL = dot(surface.normalWS, vecData.lightDirWS);  //这里没有用withRain的可能是简化细碎的雨水细节
    dotData.NdotL_XZ = dot(surface.normalWS.xz, vecData.lightDirWS.xz);  //这里没有用withRain的可能是简化细碎的雨水细节
    dotData.NdotH = dot(surface.normalWS_withRain, vecData.halfDirWS);
    dotData.NdotV = dot(surface.normalWS_withRain, vecData.viewDirWS);

    dotData.VdotH = 0;
    dotData.LdotH = 0;

    dotData.NdotL_clamped = saturate(dotData.NdotL);
    dotData.NdotV_clamped = saturate(dotData.NdotV);
    dotData.NdotV2_clamped = dotData.NdotV_clamped * dotData.NdotV_clamped;

    dotData.NgtLdotForwardWS_clamped = saturate(-dot(vecData.lightDirWS_XZ.xz, vecData.cameraForwardWS_XZ.xz));
}

void SurfaceConvertToBSDF(EndFieldVecData vecData, EndFieldDotData dotData, EndFieldSurface surface, inout EndFieldBSDF bsdf)
{
    bsdf.diffuse = ComputeDiffuseColor(surface.basecolor, surface.metallic) * 0.96f;  //0.96f is EndField Feature
    //阴影
    bsdf.diffuse_shadow = ComputeDiffuseColor(surface.basecolor_shadow, surface.metallic) * 0.96f;  //0.96f is EndField Feature
    //更深的阴影
    bsdf.diffuse_shadow2 = bsdf.diffuse_shadow * _CharacterParams0.z * 0.65f;  //0.65f is EndField Feature
    bsdf.diffuse_shadow2 = lerp(Luminance(bsdf.diffuse_shadow2), bsdf.diffuse_shadow2, 1.2f);

    bsdf.specularLevel = surface.specularLevel * DEFAULT_SPECULAR_VALUE;
    bsdf.F0 = ComputeFresnel0(surface.basecolor, surface.metallic, bsdf.specularLevel);
    bsdf.F90 = 1;
    bsdf.perceptualRoughness = surface.perceptualRoughness;
    bsdf.roughness = max(0.0078125, PerceptualRoughnessToRoughness(bsdf.perceptualRoughness));
    bsdf.anisotropy = surface.anisotropy;
    ConvertAnisotropyToRoughness(bsdf.perceptualRoughness, bsdf.anisotropy, bsdf.roughnessT, bsdf.roughnessB);
    bsdf.roughness2 = bsdf.roughnessT * bsdf.roughnessT;
    bsdf.occlusion = surface.occlusion;

    bsdf.ramp_NdotL = SampleRampColor_NdotL(vecData, dotData);
    bsdf.ramp_NdotV = SampleRampColor_NdotV(vecData, surface);
}

//region 环境镜面反射
float3 EnvBRDF(EndFieldDotData dotData, EndFieldSurface surface, EndFieldBSDF bsdf)
{
    // 这部分代码实现了环境光照的双向反射分布函数(BRDF)的积分近似。
    // 它使用了高精度的有理函数拟合(Rational Fit)来替代常用的 LUT 采样。
    // 同时包含了一个多重散射(Multi-Scatter)的修正项，用于保持高粗糙度下的能量守恒

    // 预计算粗糙度和 NdotV 的幂次，用于后续的多项式计算
    float envBRDF_roughness2 = surface.perceptualRoughness * surface.perceptualRoughness;
    float rain_NdotV3 = dotData.NdotV2_clamped * dotData.NdotV_clamped;                   //r14.z,     r14.x: rain_NdotV2, r22.x: rain_NdotV
    float envBRDF_roughness6 = envBRDF_roughness2 * envBRDF_roughness2 * envBRDF_roughness2;

    float const1 = 1.0f;

    // 2. 计算 DFG Scale (A) - 对应 F0 的缩放系数
    // 公式形式为有理函数: P(x, y) / Q(x, y)
    // 系数看起来是特定渲染管线（如米哈游 HGRP 或定制管线）预计算的拟合参数
    float termA_Num_X = dot(float2(3.327069997f, const1), float2(dotData.NdotV_clamped, 0.0365463f));
    float termA_Num_Y = dot(float2(-9.047559738f, const1), float2(dotData.NdotV_clamped, 9.0632f));
    float termA_Num   = dot(float2(termA_Num_X, termA_Num_Y), float2(const1, envBRDF_roughness2));

    float termA_Den_X = dot(float3(3.596849918f, -1.367720007f, const1), float3(dotData.NdotV2_clamped, rain_NdotV3, const1));
    float termA_Den_Y = dot(float3(-16.317399978f, const1, 9.229490280f), float3(dotData.NdotV2_clamped, 9.04401f, rain_NdotV3));

    float termA_Den_Z = dot(float3(1.0f, 19.788600921f, -20.212299346f), float3(5.56589f, dotData.NdotV2_clamped, rain_NdotV3));
    float termA_Den   = dot(float3(termA_Den_X, termA_Den_Y, termA_Den_Z), float3(const1, envBRDF_roughness2, envBRDF_roughness6));

    float dfgScale = termA_Num / termA_Den; // 变量 A

    // 3. 计算 DFG Bias (B) - 对应 F0 的偏移系数
    float termB_Num_X = dot(float2(-1.285140037f, const1), float2(dotData.NdotV_clamped, 0.99044f));
    float termB_Num_Y = dot(float2(const1, -0.755906999f), float2(1.29678f, dotData.NdotV_clamped));
    float termB_Num   = dot(float2(termB_Num_X, termB_Num_Y), float2(const1, envBRDF_roughness2));

    float termB_Den_X = dot(float3(2.923379898f, 59.418800354f, const1), float3(dotData.NdotV_clamped, rain_NdotV3, const1));
    float termB_Den_Y = dot(float3(const1, -27.030199050f, 222.591995239f), float3(20.3225f, dotData.NdotV_clamped, rain_NdotV3));
    float termB_Den_Z = dot(float3(626.130004882f, 316.627014160f, const1), float3(dotData.NdotV_clamped, rain_NdotV3, 121.563f));
    float termB_Den   = dot(float3(termB_Den_X, termB_Den_Y, termB_Den_Z), float3(const1, envBRDF_roughness2, envBRDF_roughness6));

    float dfgBias = termB_Num / termB_Den; // 变量 B

    // 4. 计算单次散射高光 (Single Scattering Specular)
    // Standard Split-Sum approximation: Specular = F0 * Scale + Bias
    float3 singleScatterSpecular = bsdf.F0 * dfgScale + dfgBias; // 对应 color_266_123

    // 5. 多重散射能量补偿 (Multi-Scattering Energy Compensation)
    // 在高粗糙度下，单次散射会丢失能量导致边缘变暗。这里进行经验性的补偿。
    // integratedEnergy = Scale + Bias 近似代表了对于白色 F0 的总反射能量积分 (Integrated BRDF for F0=1)
    float integratedEnergy = dfgScale + dfgBias;

    // 计算缺失的能量比例，并归一化
    // (1 - E) / E 是一种常见的用于推导多重散射因子的形式
    // 原始逻辑: multiScatterFactor = (1.0f - integratedEnergy) / integratedEnergy;
    float multiScatterFactor = (1.0f - integratedEnergy) / integratedEnergy;

    // 最终组合：
    // EnvBRDF = SingleScatter + MultiScatter
    // MultiScatter term = F0 * multiScatterFactor * SingleScatter
    // 这种组合方式类似于 Kulla-Conty 近似的简化版：Fms = Eavg * Ems(r) ... 这里的实现看起来是：
    // Result = SingleScatter * (1 + F0 * ((1-E)/E))
    //        = SingleScatter + SingleScatter * F0 * (1/E - 1)
    //        = E_ss + E_ss * F0 * (1/E - 1)  <-- 这是一种定制的补偿混合
    float3 envBRDF = bsdf.F0 * multiScatterFactor * singleScatterSpecular + singleScatterSpecular;
    return envBRDF;
}

real PerceptualRoughnessToMipmapLevel_EndField(real perceptualRoughness, uint maxMipLevel)
{
    float mip = max(0.001, perceptualRoughness);
    mip = log2(mip);
    mip = 1 - mip * 1.2;
    mip = maxMipLevel - mip;
    return mip;
}

float3 EnvSpecular(EndFieldVecData vecData, EndFieldSurface surface)
{
    float mip = PerceptualRoughnessToMipmapLevel_EndField(surface.perceptualRoughness, 6/*EndField is 6*/);
    float3 envSpecular = SAMPLE_TEXTURECUBE_LOD(_IndirSpecCubemap, sampler_LinearRepeat, vecData.reflectDirWS, mip).xyz;
    return envSpecular;
}

//region skin shadowColorLut
#define LUT_A 0.03125f  // 1.0/32
#define LUT_B 0.0302734375f  // 31/1024
#define LUT_C 0.00048828125f  // 0.5/1024
#define LUT_D 0.96875f  // 31/32
#define LUT_E 0.015625f  // 0.5/32

//endregion

// 雨水效果核心参数（可根据需求调整）
#define RAIN_NOISE_SCALE_1 30.0f      // 噪声频率1（控制雨滴密度）
#define RAIN_NOISE_SCALE_2 45.345600128173828125f   // 噪声频率2（补充细节）
#define RAIN_TIME_SPEED_1 3.0f        // 雨滴下落速度1
#define RAIN_TIME_SPEED_2 4.345600128173828125f     // 雨滴下落速度2
#define RAIN_DROP_SIZE_MIN 0.6f       // 雨滴最小尺寸系数
#define RAIN_DROP_SIZE_MAX 1.0f       // 雨滴最大尺寸系数
#define RAIN_ROUGHNESS_WET 0.1f       // 湿润区域粗糙度（越小越光滑）
#define RAIN_METALLIC_WET 1.0f        // 湿润区域金属度（越大反射越强）

//region rain feature func

// 生成伪随机数（输入2D坐标，输出0~1随机值）
float2 GenerateRandom2D(float2 gridPos)
{
    // 随机数种子（固定值，保证随机性）
    float2 randSeed = float2(123.339996337890625f, 456.209991455078125f);
    float2 gridRand = frac(gridPos * randSeed);
    // 点积强化随机性
    gridRand += dot(gridRand, gridRand + 34.345001220703125f).xx;
    return frac(float2(gridRand.x * gridRand.y, gridRand.x + gridRand.y));
}

// 计算单组雨滴强度（输入：坐标、时间系数、尺寸系数，输出：雨滴强度+偏移）
float2 CalculateRainDropIntensity(float2 coord, float timeFactor, out float2 dropOffset)
{
    // 1. 划分噪声格子
    float2 gridPos = floor(coord);
    float2 gridUV = frac(coord) - 0.5f;

    // 2. 生成格子内随机值
    float2 randVal1 = GenerateRandom2D(gridPos);
    float2 randVal2 = GenerateRandom2D(gridPos + 114.51399993896484375f); // 偏移格子避免重复

    // 3. 雨滴中心随机偏移
    dropOffset = ((randVal2 * 2.0f - 1.0f) * 0.25f) + (coord - gridPos) - 0.5f;
    // 雨滴形状非对称拉伸（模拟下落轨迹）
    dropOffset.x *= 1.25f;
    dropOffset.y *= dropOffset.y < 0.0f ? 1.25f : 0.75f;

    // 4. 雨滴范围判断（距离中心越近强度越高）
    float dropLength = length(dropOffset);
    float dropSize = 0.25f * lerp(0.60f, 1.0f, randVal1.x);
    float rangeIntensity = smoothstep(dropSize, 0.0f, dropLength);

    // 5. 时间窗口判断（模拟雨滴下落的生命周期）
    float timeCycle = frac(timeFactor + randVal1.x);
    float2 timeIntensity;
    timeIntensity.x = smoothstep(0.2f, 0.22f, timeCycle) * smoothstep(0.5f, 0.2f, timeCycle);


    // 6. 最终雨滴强度（范围+时间双重过滤）
    timeIntensity.x *= step(0.001f, rangeIntensity);
    timeIntensity.y = randVal2.y;

    //输出
    dropOffset = clamp(dropOffset / dropSize, -1.0f, 1.0f) * timeIntensity.x * lerp(0.5f, 1.0f, randVal2.x);
    // return float2(dropLength, 0);
    return timeIntensity;
}

//cbt-3 skin
void ApplyRainFeature(VecData vecData, float4 _WaterParam, float perceptualRoughness, float metallicRatio,
    inout float rainIntensity, inout float wetPerceptualRoughness, inout float wetMetallicRatio, inout float3 wetNormalWS)
{
    // float4 _WaterParam = _20_m0_12_m7;
    float _PosScale = _cb0_170.z;
    // float baseMetallicRatio = _46_m1;

    float depthDiff = _WaterParam.z - vecData.positionWS.y;
    float posRainIntensity = smoothstep(-0.2f, 0.15f, depthDiff) * _WaterParam.y;
    float enableRain = _WaterParam.x + posRainIntensity;

    if (9.99999975e-005 < enableRain)
    {
        // 最终雨水强度权重（取全局和位置的最大值）
        float finalRainWeight = max(_WaterParam.x, posRainIntensity);

        bool switchDir = _RainFilpObjectDir > 0.01f;

        float3 posOS = vecData.positionOS;
        posOS = switchDir ? posOS.xzy * float3(1.0f, 1.0f, -1.0f) : posOS.xyz;
        float3 scaledPosOS = posOS * _PosScale;

        float3 normalAbs = abs(vecData.normalWS_raw);
        normalAbs = switchDir ? normalAbs.xzy : normalAbs.xyz;
        float3 normalWeight = max(pow(normalAbs - 0.2f, 10.0f), 0.0f);
        // 归一化权重，确保三个轴权重和为1
        float3 normalWeightNLZ = normalWeight / dot(normalWeight, float3(1.0f, 1.0f, 1.0f));

        // 时间因子（控制雨滴下落动画）
        float timeFactor1 = _Time.x * RAIN_TIME_SPEED_1;
        float timeFactor2 = _Time.x * RAIN_TIME_SPEED_2;

        // 存储三组雨滴的强度和偏移
        float2 dropIntensity[6];
        float2 dropOffset[6];

        // 第一组：XZ轴（频率1）
        dropIntensity[0] = CalculateRainDropIntensity(scaledPosOS.xz * RAIN_NOISE_SCALE_1, timeFactor1, dropOffset[0]);
        // 第二组：XY轴（频率1）
        dropIntensity[1] = CalculateRainDropIntensity(scaledPosOS.xy * RAIN_NOISE_SCALE_1, timeFactor1, dropOffset[1]);
        // 第三组：ZY轴（频率1）
        dropIntensity[2] = CalculateRainDropIntensity(scaledPosOS.zy * RAIN_NOISE_SCALE_1, timeFactor1, dropOffset[2]);
        // 第四~六组：补充频率2的细节
        dropIntensity[3] = CalculateRainDropIntensity(scaledPosOS.xz * RAIN_NOISE_SCALE_2, timeFactor2, dropOffset[3]);
        dropIntensity[4] = CalculateRainDropIntensity(scaledPosOS.xy * RAIN_NOISE_SCALE_2, timeFactor2, dropOffset[4]);
        dropIntensity[5] = CalculateRainDropIntensity(scaledPosOS.zy * RAIN_NOISE_SCALE_2, timeFactor2, dropOffset[5]);
        // 计算雨水流动的切线方向（与世界Y轴垂直）
        float3 tangentDir = cross(vecData.normalWS, float3(0.0f, 1.0f, 0.0f));
        float3 flowDir = length(tangentDir) > 1e-4f ? normalize(tangentDir) : float3(-1.0f, 0.0f, 0.0f);

        // 合并三组雨滴的偏移，按法线权重加权
        //xy
        float2 part1DropOffset = dropOffset[0] * normalWeightNLZ.y + dropOffset[1] * normalWeightNLZ.z + dropOffset[2] * normalWeightNLZ.x;
        float2 part2DropOffset = dropOffset[3] * normalWeightNLZ.y + dropOffset[4] * normalWeightNLZ.z + dropOffset[5] * normalWeightNLZ.x;
        //zw
        float2 part1Intensity = dropIntensity[0] * normalWeightNLZ.y + dropIntensity[1] * normalWeightNLZ.z + dropIntensity[2] * normalWeightNLZ.x;
        float2 part2Intensity = dropIntensity[3] * normalWeightNLZ.y + dropIntensity[4] * normalWeightNLZ.z + dropIntensity[5] * normalWeightNLZ.x;

        float2 totalIntensity = max(part1Intensity, part2Intensity);
        float2 totalDropOffset = part1DropOffset + part2DropOffset;
        // 最终雨水强度（结合全局开关）
        rainIntensity = totalIntensity.x * step(1.0f - _WaterParam.x, totalIntensity.y - 0.1f);
        bool2 cmp_Intensity = rainIntensity > 0.001f;
        float2 rainDropOffset = float2(cmp_Intensity.x ? totalDropOffset.x : 0.0f, cmp_Intensity.y ? totalDropOffset.y : 0.0f);

        // 法线扰动：原始法线 插值到 流动方向法线
        float3 perturbedNormal = lerp(vecData.normalWS, flowDir, rainDropOffset.x);
        perturbedNormal = normalize(lerp(perturbedNormal, cross(vecData.normalWS, flowDir), rainDropOffset.y));
        wetNormalWS = lerp(vecData.normalWS, perturbedNormal, rainIntensity);

        // 粗糙度插值：基础粗糙度 -> 半湿润 -> 完全湿润
        float roughHalfWet = lerp(perceptualRoughness, 0.5f, finalRainWeight);
        wetPerceptualRoughness = lerp(roughHalfWet, RAIN_ROUGHNESS_WET, rainIntensity);

        // 金属度插值：基础金属度 -> 半湿润 -> 完全湿润
        float metallicHalfWet = lerp(metallicRatio, RAIN_METALLIC_WET, finalRainWeight);
        wetMetallicRatio = lerp(metallicHalfWet, RAIN_METALLIC_WET, rainIntensity);
    }
    else
    {
        rainIntensity = 0;
        perceptualRoughness = perceptualRoughness + 0.0f;
        metallicRatio = metallicRatio + 0.0f;
        wetNormalWS = vecData.normalWS;
    }
}
//endregion

void InitialVecData(Varyings i, uint facing, float3 bumpTS, inout VecData vecData)
{
    vecData.positionHCS = i.positionHCS;
    vecData.positionWS = i.positionWS;
    vecData.normalWS_raw = SafeNormalize(i.normalWS.xyz);
    vecData.bitangentWS_raw = SafeNormalize(i.bitangentWS.zxy);
    vecData.normalWS_raw = facing > 0 ? vecData.normalWS_raw : -vecData.normalWS_raw;
    float3x3 TBN = float3x3(i.tangentWS.xyz, i.bitangentWS.xyz, vecData.normalWS_raw.xyz);

    vecData.normalWS = SafeNormalize(TransformTangentToWorld(bumpTS, TBN));
    vecData.bitangentWS = normalize(vecData.bitangentWS_raw.zxy);
    vecData.tangentWS = cross(vecData.bitangentWS, vecData.normalWS);
    vecData.bitangentWS = cross(vecData.normalWS, vecData.tangentWS);

    vecData.viewDirWS = SafeNormalize(i.viewDirWS.xyz);
    vecData.reflectDirWS = reflect(vecData.viewDirWS, vecData.normalWS);
}

void InitialDotData(VecData vecData, inout DotData dotData)
{
    float ndotl = dot(vecData.normalWS, vecData.lightDirWS);
    dotData.ndl = max(0, ndotl);
    float ndoth = dot(vecData.normalWS, vecData.halfDirWS);
    dotData.ndh = max(0, ndoth);
    float vdoth = dot(vecData.viewDirWS, vecData.halfDirWS);
    dotData.vdh = max(0, vdoth);
    #ifdef _SHADOW_RAMP
    float ldoth = dot(vecData.lightDirWS, vecData.halfDirWS);    //dot(lightDirWS, halfDir)
    dotData.ldh = max(0, ldoth);
    #endif
}

//region fog feature
static const float K_LOG2E = 1.44269502162933349609375f; // 1/ln(2)
static const float K_PI = 3.14159265359f;

// 辅助函数：计算高度雾的指数积分
// 对应: (1 - exp(-falloff * dist)) / falloff * exp(-falloff * (camY - base))
// 参数说明：
// vectorY:   视线向量的 Y 分量 (posWS.y - camPos.y)
// camHeight: 相机世界高度
// baseHeight: 雾起始高度 (_ExponentialFogParams.x 或 .z)
// falloff:    高度衰减系数 (_ExponentialFogParams.z 或 .x)
// density:    雾密度 (_ExponentialFogParams.y)
float ComputeHeightFogIntegral(float vectorY, float camHeight, float baseHeight, float falloff, float density)
{
    // --- 1. 计算相机处的指数项 (Camera Term) ---
    // Text 2: Line 196-200
    // exponent = (CamY - Base) * Falloff
    float camExponent = (camHeight - baseHeight) * falloff;
    camExponent = max(camExponent, -127.0f); // 保护 exp2

    // camTerm = Density * 2^(-exponent)
    float camTerm = density * exp2(-camExponent);

    // --- 2. 计算积分变量 x (Delta Height Term) ---
    // Text 2: Line 201
    // x = VectorY * Falloff
    float x = vectorY * falloff;

    // --- 3. 计算准确解 (Exact Solution) ---
    // Text 2: Line 202-205
    // result = (1 - 2^(-x)) / x
    // 这里 text 2 也对 x 做了 max(-127) 保护，虽然主要影响 pow 的结果
    float x_safe = max(x, -127.0f);
    float expVal = exp2(-x_safe);
    float resExact = (1.0f - expVal) / x_safe; // 如果 x=0 这里会 NaN，但后面会处理

    // --- 4. 计算近似解 (Taylor Approximation) ---
    // Text 2: Line 212
    // 用于 x 接近 0 时的情况
    // Taylor: ln(2) - x * (ln(2)^2 / 2)
    // 0.6931472 = ln(2)
    // 0.2402265 = 0.480453 / 2 = (ln(2)^2) / 2
    float resApprox = 0.6931472f - x_safe * 0.2402265f;

    // --- 5. 选择分支 (Branchless Selection) ---
    // Text 2: Line 213-214
    // 阈值 5.96e-08f (即 0.00000006)
    float finalInt = (abs(x_safe) > 5.96e-08f) ? resExact : resApprox;

    // --- 6. 组合结果 ---
    // Text 2: Line 215
    return camTerm * finalInt;
}

// =================================================================================
// 主雾效逻辑
// =================================================================================
float3 ApplyFog(float3 sceneColor, float3 posWS, float3 screenPos)
{
    return 0;
}

#endif
