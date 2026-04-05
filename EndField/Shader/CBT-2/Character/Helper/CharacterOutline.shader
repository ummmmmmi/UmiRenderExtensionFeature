Shader "DanbaidongRP/EndFieldToon/Helpers/Outline"
{
    Properties
    {
        // Outline Props
        [FoldoutBegin(_FoldoutOutlineEnd)]_FoldoutOutline("Outline", float) = 0
            [Toggle(_USE_SMOOTH_NORMAL)]
            _OutlineColor                   ("Outline Color", Color)                = (0, 0, 0, 0.8)
            _OutlineWidth                   ("Width", Range(0, 10))                 = 1.0
            _OutlineZOffset                 ("描边深度偏移，custom描边需要默认0.001", Range(-1, 1))            = 0.0
            [Toggle(_USE_SMOOTH_NORMAL)]
            _USE_SMOOTH_NORMAL              ("使用平滑法线", float)                   = 0

            [Title(Lighting)]
            [HDR]_OutlineDirectLightingColor    ("DirectColor", color)              = (1,1,1,0.5)
            _OutlineDirectLightingOffset        ("DirectOffset", Range(-1, 1))      = 0.0
            [HDR]_OutlinePunctualLightingColor  ("PunctualColor", color)            = (1,1,1,0.5)
            _OutlinePunctualLightingOffset      ("PunctualOffset", Range(-1, 1))    = 0.0

        [FoldoutEnd]_FoldoutOutlineEnd("_FoldoutEnd", float) = 0
    }
    SubShader
    {
        Tags
        {
            "RenderType"="Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "Queue"="Geometry"
            "IgnoreProjector" = "True"
        }
        LOD 300

        // ForwardOutline
        Pass
        {
            Name "ForwardOutline"

            Tags
            {
                "LightMode" = "CharacterOutline"
            }

            Cull Front
            ZWrite On

            HLSLPROGRAM
            #pragma target 4.5

            // -------------------------------------
            // Shader Stages
            #pragma vertex ToonOutlineVert
            #pragma fragment ToonOutlineFrag

            // -------------------------------------
            // Material Keywords
            #pragma shader_feature_local _USE_SMOOTH_NORMAL SN_VertNormal
            // -------------------------------------
            // Universal Pipeline keywords
            // #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE


            //#pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile _ _GPU_LIGHTS_CLUSTER
            // #pragma multi_compile_fragment _ _REFLECTION_PROBE_BLENDING
            // #pragma multi_compile_fragment _ _REFLECTION_PROBE_BOX_PROJECTION
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile_fragment _ _DBUFFER_MRT1 _DBUFFER_MRT2 _DBUFFER_MRT3
            #pragma multi_compile_fragment _ _RENDER_PASS_ENABLED
            #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/RenderingLayers.hlsl"

            // -------------------------------------
            // Unity defined keywords
            // #pragma multi_compile _ LIGHTMAP_SHADOW_MIXING
            // #pragma multi_compile _ SHADOWS_SHADOWMASK
            // #pragma multi_compile _ DIRLIGHTMAP_COMBINED
            // #pragma multi_compile _ LIGHTMAP_ON
            // #pragma multi_compile _ DYNAMICLIGHTMAP_ON
            #pragma multi_compile_fragment _ LOD_FADE_CROSSFADE
            #pragma multi_compile_fragment _ _GBUFFER_NORMALS_OCT

            //--------------------------------------
            // GPU Instancing
            #pragma multi_compile_instancing
            #pragma instancing_options renderinglayer
            #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/DOTS.hlsl"

            // -------------------------------------
            // Includes
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/UnityInstancing.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/UnityGBuffer.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/GPUCulledLights.hlsl"


            CBUFFER_START(UnityPerMaterial)
            float4 _BaseMap_ST;
            float4 _OutlineColor;
            float _OutlineWidth;
            float _OutlineZOffset;

            // Lighting
            float4 _OutlineDirectLightingColor;
            float _OutlineDirectLightingOffset;
            float4 _OutlinePunctualLightingColor;
            float _OutlinePunctualLightingOffset;

            float _AlphaClip;
            CBUFFER_END

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            struct Attributes
            {
                float4 vertex   :POSITION;
                float3 normal   :NORMAL;
                float4 tangent  :TANGENT;
                float4 color    :COLOR;
                float4 uv       :TEXCOORD0;
                float4 uv1       :TEXCOORD1;
                float4 uv2       :TEXCOORD2;
                float4 uv3       :TEXCOORD3;
                float4 uv4       :TEXCOORD4;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionHCS   :SV_POSITION;
                float3 positionWS   :TEXCOORD0;
                float3 normalWS     :TEXCOORD1;
                float4 uv           :TEXCOORD2;
                float4 color        :TEXCOORD3;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };

            float3 OctahedronToUnitVector( float2 Oct )
            {
                float3 N = float3(Oct.x, Oct.y, 1 - abs(Oct.x) - abs(Oct.y));
                float2 signOct = sign(Oct);
                float isBackface = step(0.0, -N.z);
                N.xy = lerp(N.xy, (1 - abs(N.yx)) * signOct, isBackface);
                return normalize(N);
            }

            float3 HemiOctahedronToUnitVector( float2 Oct )
            {
                float2 val = float2(Oct.x + Oct.y, Oct.x - Oct.y) * 0.5;
                float3 N = float3(val.x, val.y, 1 - dot(abs(val), 1.0));

                return normalize(N);
            }

            float3 SphericalToUnitVector(float2 Oct)
            {
                float3 N;
                N.xy = Oct;
                N.z = sqrt(1 - min(1.0f, dot(N.xy, N.xy)));
                return N;
            }

            float4 UseNewGFOutline(Attributes IN, VertexPositionInputs positionInputs, VertexNormalInputs normalInputs, float4 vertColor)
            {
                float vertColor_r = vertColor.r;
                float vertColor_g = 1;
                float3 smoothNormalWS = normalInputs.normalWS;

                float3 smoothNormalVS = TransformWorldToViewNormal(smoothNormalWS);
                float3 positionVS = positionInputs.positionVS;

                float4 positionVS_raw = float4(positionVS, 1.0);
                float4 positionVS_Offset = float4(positionVS + smoothNormalVS * 0.00052083336 * _OutlineWidth, 1.0);

                float3 matrix_P0 = float3(UNITY_MATRIX_P[0][0], UNITY_MATRIX_P[0][2], UNITY_MATRIX_P[0][3]);
                float3 matrix_P1 = float3(UNITY_MATRIX_P[1][1], UNITY_MATRIX_P[1][2], UNITY_MATRIX_P[1][3]);
                float4 matrix_P2 = float4(UNITY_MATRIX_P[2][0], UNITY_MATRIX_P[2][1], UNITY_MATRIX_P[2][2], UNITY_MATRIX_P[2][3]);
                float2 matrix_P3 = float2(UNITY_MATRIX_P[3][2], UNITY_MATRIX_P[3][3]);

                float x_raw   = dot(matrix_P0, positionVS_raw.xzw);
                float x_offset = dot(matrix_P0, positionVS_Offset.xzw);

                float y_raw   = dot(matrix_P1, positionVS_raw.yzw);
                float y_offset = dot(matrix_P1, positionVS_Offset.yzw);

                float4 positionCS;
                positionCS.xy = float2(x_offset, y_offset);

                float2 delta = positionCS.xy - float2(x_raw, y_raw);
                delta *= 1.3 * 3;

                float3 smoothNormalCS = TransformWorldToHClipDir(smoothNormalWS, true);

                float4 temp;
                temp.xy = int2(-1, -1) + _ScreenParams.zw;
                temp.xy *= smoothNormalCS.xy;
                temp.xy += temp.xy;

                positionCS.w = dot(matrix_P3, float2(positionVS.z, 1.0));

                temp.zw = temp.xy * positionCS.w;
                temp.xy = _ScreenParams.x * temp.xy * positionCS.w;
                temp = float4(0.00130208337, 0.00130208337, 1.2, 1.2) * temp;

                float2 zw_abs = abs(temp.zw);
                float2 xy_abs = abs(temp.xy);
                float2 delta_abs = abs(delta);
                float2 max_val = max(zw_abs, delta_abs);
                float2 min_val = min(max_val, xy_abs);

                temp.xy = smoothNormalCS.xy < float2(0.0, 0.0) ? int2(1, 1) : int2(0, 0);
                temp.zw = smoothNormalCS.xy > float2(0.0, 0.0) ? int2(1, 1) : int2(0, 0);
                float2 r0 = temp.zw - temp.xy;

                positionCS.xy = r0 * min_val * vertColor_r + float2(x_raw, y_raw);

                positionCS.z = dot(matrix_P2, float4(positionVS, 1.0));
                positionCS.z = positionCS.z - _OutlineZOffset * 0.001 * vertColor_g;

                return positionCS;
            }

            float4 UseCustomOutline(Attributes IN, VertexPositionInputs positionInputs, VertexNormalInputs normalInputs, float4 vertColor)
            {
                float color_g = 1;
                float color_r = 1;

                float3 normalVS = TransformWorldToViewDir(normalInputs.normalWS);
                normalVS.z = 0.00001;
                normalVS = normalize(normalVS);

                float r1 = rcp(UNITY_MATRIX_P[1][1]);
                float r2 = positionInputs.positionVS.z * r1;
                float3 r5 = saturate(r2 * float3(10,1,0.5) + float3(0, -0.1, -1.05));

                r2 = lerp(0.00104999996 * r5.x, 0.005, r5.y);
                r2 = lerp(r2, 0.00350000011, r5.z);
                r2 = _OutlineWidth * color_r * r2;
                normalVS.xy = normalVS.xy * r2;

                float3 positionVS;

                positionVS.xy = normalVS.xy + positionInputs.positionVS.xy;
                positionVS.z = positionInputs.positionVS.z;

                float3 positionVS_normalize = normalize(positionVS);

                positionVS = unity_OrthoParams.w == 0.0
                   ? positionVS + positionVS_normalize * _OutlineZOffset * color_g
                   : positionVS - half3(0.0f, 0.0f, _OutlineZOffset * color_g);

                return TransformWViewToHClip(positionVS);
            }

            Varyings ToonOutlineVert(Attributes v)
            {
                Varyings o = (Varyings)0;

                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                VertexPositionInputs positionInputs = GetVertexPositionInputs(v.vertex.xyz);

                float3 smoothnormalTS = SphericalToUnitVector(v.uv3.xy);

                VertexNormalInputs normalInputs = GetVertexNormalInputs(v.normal.xyz);
                float3 normalWS_raw = normalInputs.normalWS;

                float3 bitangentOS = cross(v.normal, v.tangent.xyz) * v.tangent.w * GetOddNegativeScale();
                float3x3 TBNOS = float3x3(v.tangent.xyz, bitangentOS, v.normal.xyz);
                float3 smoothnormalOS = mul(smoothnormalTS, TBNOS);

                normalInputs.normalWS = TransformObjectToWorldNormal(smoothnormalOS);

                o.positionHCS = UseNewGFOutline(v, positionInputs, normalInputs, v.color);  //少前2做法CS描边，任何距离自适应描边宽度，基本只有深度偏移可调
                // o.positionHCS = UseCustomOutline(v, positionInputs, normalInputs, v.color);  //结合战双、绝区零做法VS描边，描边宽度、深度偏移可调，补充正交、透视相机宽度适配，自由度更好，但细微效果不及少前2的

                o.normalWS = normalWS_raw;

                o.uv.xy = TRANSFORM_TEX(v.uv,_BaseMap);

                o.color = float4(smoothnormalOS, 1);
                return o;
            }

            float4 ToonOutlineFrag(Varyings i) : SV_Target0
            {
                UNITY_SETUP_INSTANCE_ID(i);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                // Alpha Clip
                float4 mainTexColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv.xy);
            #if _USE_ALPHA_CLIPPING
                clip(mainTexColor.a - _AlphaClip);
            #endif

                float3 oulineColor = lerp(mainTexColor.xyz, _OutlineColor.rgb, _OutlineColor.a);

                // Input
                float  depth = i.positionHCS.z;
                float3 positionWS = i.positionWS;
                float2 screenUV = GetNormalizedScreenSpaceUV(i.positionHCS.xy);

                // Property prepare
                float3 normalWS = SafeNormalize(i.normalWS);
                float3 normalVS = TransformWorldToViewNormal(normalWS);
                float directThreshold = saturate(0.8 - _OutlineDirectLightingOffset);
                float punctualThreshold = saturate(0.8 - _OutlinePunctualLightingOffset);

                float3 directLighting = 0;
                float3 punctualLighting = 0;

                // Direct Outline Lighting
                uint dirLightIndex = 0;
                for (dirLightIndex = 0; dirLightIndex < _DirectionalLightCount; dirLightIndex++)
                {
                    DirectionalLightData dirLight = g_DirectionalLightDatas[dirLightIndex];

                    #ifdef _LIGHT_LAYERS
                    if (IsMatchingLightLayer(dirLight.lightLayerMask, meshRenderingLayers))
                    #endif
                    {
                        float3 lightDirWS = dirLight.lightDirection;
                        float3 lightDirVS = TransformWorldToViewDir(lightDirWS);
                        // float NdotL = dot(normalVS.xy, lightDirVS.xy);
                        float NdotL = dot(normalWS, -lightDirWS);

                        float3 lightColor = lerp(_OutlineDirectLightingColor.rgb, _OutlineDirectLightingColor.rgb * dirLight.lightColor,  _OutlineDirectLightingColor.a);

                        // directLighting += step(directThreshold, NdotL) * lightColor;
                        directLighting += NdotL * lightColor * oulineColor * saturate(_OutlineDirectLightingOffset);
                    }
                }
                // TODO: Apply Shadow
                float4 shadowCoord = TransformWorldToShadowCoord(positionWS);
                float shadowAttenuation = MainLightRealtimeShadow(shadowCoord);
                directLighting *= shadowAttenuation;


                // Punctual Outline Lighting
                uint lightCategory = LIGHTCATEGORY_PUNCTUAL;
                uint lightStart;
                uint lightCount;
                PositionInputs posInput = GetPositionInput(i.positionHCS.xy, _ScreenSize.zw, depth, UNITY_MATRIX_I_VP, UNITY_MATRIX_V);
                GetCountAndStart(posInput, lightCategory, lightStart, lightCount);
                uint v_lightListOffset = 0;
                uint v_lightIdx = lightStart;

                if (lightCount > 0) // avoid 0 iteration warning.
                {
                    while (v_lightListOffset < lightCount)
                    {
                        v_lightIdx = FetchIndex(lightStart, v_lightListOffset);
                        if (v_lightIdx == -1)
                            break;

                        GPULightData gpuLight = FetchLight(v_lightIdx);

                        #ifdef _LIGHT_LAYERS
                        if (IsMatchingLightLayer(gpuLight.lightLayerMask, meshRenderingLayers))
                        #endif
                        {
                            float3 lightVector = gpuLight.lightPosWS - positionWS.xyz;
                            float distanceSqr = max(dot(lightVector, lightVector), FLT_MIN);
                            float3 lightDirection = float3(lightVector * rsqrt(distanceSqr));
                            float shadowMask = 1;

                            float distanceAtten = DistanceAttenuation(distanceSqr, gpuLight.lightAttenuation.xy) * AngleAttenuation(gpuLight.lightDirection.xyz, lightDirection, gpuLight.lightAttenuation.zw);
                            float shadowAtten = gpuLight.shadowType == 0 ? 1 : AdditionalLightShadow(gpuLight.shadowLightIndex, positionWS, lightDirection, shadowMask, gpuLight.lightOcclusionProbInfo);
                            float attenuation = distanceAtten * shadowAtten;

                            float NdotL = dot(normalWS, lightDirection);

                            float3 lightColor = lerp(_OutlinePunctualLightingColor.rgb, _OutlinePunctualLightingColor.rgb * gpuLight.lightColor,  _OutlinePunctualLightingColor.a);

                            punctualLighting += lightColor * step(punctualThreshold, NdotL) * attenuation * gpuLight.outlineContribution;
                        }

                        v_lightListOffset++;
                    }
                }

                float3 result = directLighting + punctualLighting;
                result = AcesTonemap(result);

                return float4(result, 1);
            }

            ENDHLSL
        }

    }

    CustomEditor "UnityEditor.DanbaidongGUI.DanbaidongGUI"
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
