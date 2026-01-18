Shader "WangYue/PostProcess/NegativeColor"
{
    Properties
    {
        // This property is necessary to make the CommandBuffer.Blit bind the source texture to _MainTex
        _MainTex("Main Texture", 2DArray) = "grey" {}
    	
    	_NegativeRGBOffset("负片 RGB Hue偏移", Vector) = (1,1,1,0)
    	_NegativeSphereParams("负片 球心位置/半径", Vector) = (0,0,0,1)
    	_NegativeSphereFadeParams("负片 球空间边缘范围/边缘过渡", Vector) = (0.5,0.5,0,0)
    	
        
        // _BlurCenter           ("BlurCenter",           Vector) = (1,1,1,1)
		// _RadialBlurParameter  ("RadialBlurParameter",  Vector) = (1,1,1,1)
		// _RadialStretching("径向拉伸强度", Range(0,1)) = 0
		// _RadialStretchRange("径向拉伸范围", Range(0,2)) = 1
    }

    HLSLINCLUDE
    
    #pragma target 4.5
    #pragma only_renderers d3d11 playstation xboxone xboxseries vulkan metal switch

    #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
    #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"
    #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/Core.hlsl"
    // #include "Packages/com.unity.render-pipelines.high-definition/Runtime/ShaderLibrary/ShaderVariables.hlsl"
    // #include "Packages/com.unity.render-pipelines.high-definition/Runtime/PostProcessing/Shaders/FXAA.hlsl"
    // #include "Packages/com.unity.render-pipelines.high-definition/Runtime/PostProcessing/Shaders/RTUpscale.hlsl"

 //    #define LOOPCOUNT _RadialBlurParameter.z
	// #pragma multi_compile_local _ _RADIALINVERT
	// #pragma multi_compile_local _ _USEDEPTHMASK

    struct Attributes
    {
        uint vertexID : SV_VertexID;
        UNITY_VERTEX_INPUT_INSTANCE_ID
    };

    struct Varyings
    {
        float4 positionCS : SV_POSITION;
        float2 texcoord   : TEXCOORD0;
    	float3 rayWS : TEXCOORD1; // 世界空间射线方向（逐顶点计算，插值到像素）
        UNITY_VERTEX_OUTPUT_STEREO
    };

    Varyings Vert(Attributes input)
    {
        Varyings output;
        UNITY_SETUP_INSTANCE_ID(input);
        UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
        output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
        output.texcoord = GetFullScreenTriangleTexCoord(input.vertexID);

    	// 计算从相机穿过当前像素的射线方向
    	float2 uv = (output.texcoord.xy);
        // NDC坐标：UV(0~1) → NDC(-1~1)，注意Y轴不需要翻转（在片段着色器处理）
        float2 ndc = uv * 2.0 - 1.0;
        // 反投影到视空间（近平面上）
        float4 viewPos = mul(UNITY_MATRIX_I_P, float4(ndc, 1.0, 1.0));
        viewPos /= viewPos.w;
        // 转换到世界空间作为射线方向
        output.rayWS = mul(UNITY_MATRIX_I_V, float4(viewPos.xyz, 0.0)).xyz;
        output.rayWS = normalize(output.rayWS);
        return output;
    }

    // List of properties to control your post process effect
    TEXTURE2D_X(_MainTex);
    TEXTURE2D_X(_BlitTexture);
    SAMPLER(s_linear_clamp_sampler);
    TEXTURE2D_X(_DepthTex);
    SAMPLER(sampler_DepthTex);
    float4 _DepthTex_TexelSize;

    CBUFFER_START(UnityPerMaterial)
		float4 _NegativeRGBOffset;    // x: r; y: g; z: b; w: 色相偏移 0-1
		float4 _NegativeSphereParams;    // xyz: 位置; w: 半径
		float4 _NegativeSphereFadeParams;	//x: fadeMin, y: fadeMax
    
	CBUFFER_END

 //    float SphereFadeWS(float2 fadeParams, float3 centerWS, float radius, float2 positionNDC, float2 uv)
	// {
	// 	float depth = SampleCameraDepth(uv);
	// 	float3 posWS = ComputeWorldSpacePosition(positionNDC, depth, UNITY_MATRIX_I_VP);
	// 	float sphereMaskWS = length(posWS - centerWS.xyz) - radius;
	// 	sphereMaskWS = 1 - smoothstep(fadeParams.x, fadeParams.y, sphereMaskWS);
	// 	return sphereMaskWS;
	// }

    float SphereFadeVS(float2 fadeParams, float3 centerWS, float radius, float2 positionNDC, float2 uv)
	{
		float depth = SAMPLE_TEXTURE2D_X(_DepthTex, sampler_DepthTex, uv).r;		// depth 为 0.21左右
		// float depth = SampleCameraDepth(uv);
		float3 posVS = ComputeWorldSpacePosition(positionNDC, depth, UNITY_MATRIX_I_VP);
		float3 centerVS = mul(UNITY_MATRIX_I_V, float4(centerWS, 1)).xyz;
		float sphereMaskWS = length(posVS - centerVS.xyz) - radius;
		sphereMaskWS = 1 - smoothstep(fadeParams.x, fadeParams.y, sphereMaskWS);
		return sphereMaskWS;
	}

    float SphereFadeSS(float2 fadeParams, float2 centerSS, float radius, float2 positionNDC)
	{
		float sphereMaskSS = length(positionNDC - centerSS) - radius;
		sphereMaskSS = 1 - smoothstep(fadeParams.x, fadeParams.y, sphereMaskSS);
		return sphereMaskSS;
	}

    void CalculateViewSpaceSphereProperty(float3 centerWS, float radiusWS, out float3 outCenterPosVS, out float outRadiusVS)
	{
		//centerPos world to view
		float4 centerVS = mul(UNITY_MATRIX_V, float4(centerWS, 1));
		float3 centerPosVS = centerVS.xyz / centerVS.w;
		float radiusVS = radiusWS;
 
		//far clipPlane
		float zFar = -_ProjectionParams.z;	// x: 1 (-1 flipped), y: near,     z: far,       w: 1/far
		float zNear = -_ProjectionParams.y;
		float distToFar = abs(centerPosVS.z - zFar);	//球心深度 减 farPlane深度
		float distToNear = abs(centerPosVS.z - zNear);
 
		// 分情况判定
		bool isCenterInFrustum = (centerPosVS.z >= zFar) && (centerPosVS.z <= zNear);
		if (isCenterInFrustum)
		{
			// 场景1：球心在视锥内
	        outCenterPosVS = centerPosVS;
	        outRadiusVS = radiusVS;
		}
		else if (distToFar <= radiusVS)
		{
			// 场景2：仅与远裁面相交
	        outCenterPosVS = float3(centerPosVS.x, centerPosVS.y, zFar);
	        outRadiusVS = sqrt(radiusVS * radiusVS - distToFar * distToFar);
		}
		else if (distToNear <= radiusVS)
		{
			// 场景3：仅与近裁面相交
	        outCenterPosVS = float3(centerPosVS.x, centerPosVS.y, zNear);
	        outRadiusVS = sqrt(radiusVS * radiusVS - distToNear * distToNear);
		}
		else
		{
			// 场景4：无相交，返回无效参数
	        outCenterPosVS = float3(0,0,0);	//?
	        outRadiusVS = -1.0f;
		}
	}

	bool ProjectWorldSphereToScreenCircle(
	    float3 centerWS,
	    float  radiusWS,

	    out float2 outCenterSS,   // 0~1
	    out float  outRadiusSS    // 0~1
	)
	{
	    // -------- WS -> VS --------
	    float3 centerVS = mul(UNITY_MATRIX_V, float4(centerWS, 1.0)).xyz;

		// 新增：判断球是否在相机后方（视空间Z>0），直接返回false
	    if (centerVS.z > 0.0)
	    {
	        outCenterSS = float2(0, 0);
	        outRadiusSS = 0;
	        return false;
	    }
    
	    // Unity 视空间：相机朝 -Z，看向前方
	    float distVS = length(centerVS);

	    // 相机在球内：直接全屏
	    if (distVS <= radiusWS)
	    {
	        outCenterSS = float2(0.5, 0.5);
	        outRadiusSS = 1.5; // 足够覆盖全屏
	        return true;
	    }

	    // -------- 切锥角度 --------
	    float sinTheta = radiusWS / distVS;
	    float cosTheta = sqrt(1.0 - sinTheta * sinTheta);
	    float tanTheta = sinTheta / cosTheta;

	    // -------- 投影参数 --------
	    // UNITY_MATRIX_P:
	    // m00 = cot(fovX/2)
	    // m11 = cot(fovY/2)
	    float projScaleX = UNITY_MATRIX_P._m00;
	    float projScaleY = UNITY_MATRIX_P._m11;

	    // -------- NDC 半径 --------
	    float radiusNDC_X = tanTheta * projScaleX;
	    float radiusNDC_Y = tanTheta * projScaleY;

	    float radiusNDC = max(radiusNDC_X, radiusNDC_Y);

	    // -------- 球心投影 --------
	    float4 centerHCS = mul(UNITY_MATRIX_P, float4(centerVS, 1.0));
	    float2 centerNDC = centerHCS.xy / centerHCS.w;

	    // NDC (-1~1) -> SS (0~1)
	    outCenterSS = centerNDC * 0.5 + 0.5;
	    outRadiusSS = radiusNDC * 0.5;

	    return true;
	}

    // 球体-射线相交检测（移植自原GLSL，仅语法微调）
    float SphereIntersect(float3 ro, float3 rd, float3 sphPos, float sphRadius)
    {
        float3 oc = ro - sphPos;
        float b = dot(oc, rd);
        float c = dot(oc, oc) - sphRadius * sphRadius;
        float h = b * b - c;
        if (h < 0.0) return -1.0;
        return -b - sqrt(h); // 返回近交点距离
    }

    // 核心：从投影矩阵反推相机垂直FOV（无需外部传递）
    float GetCameraFOVFromProjMatrix(float4x4 proj)
    {
        // 1. 透视投影矩阵的关键值：proj[1][1] = 1/tan(FOV/2)
        float cotHalfFov = proj[1][1]; 
        // 2. 反推半FOV（弧度），再转角度
        float halfFovRad = atan(1.0 / cotHalfFov);
        float fovDeg = degrees(2.0 * halfFovRad);
        // 3. 容错：正交相机无FOV，返回默认60°
        return (cotHalfFov == 0.0) ? 60.0 : fovDeg;
    }

    // 计算球体在屏幕上的真实投影半径
    float GetSphereScreenRadius(float worldRadius, float camDist, float4x4 proj, float screenHeight)
    {
		float fov = GetCameraFOVFromProjMatrix(proj);
        float tanHalfFov = tan(radians(fov) / 2.0);
        float ndcRadius = tanHalfFov * worldRadius / camDist;
        float screenRadius = ndcRadius * (screenHeight / 2.0);
        return abs(screenRadius);
    }

    // 世界空间→屏幕空间（适配Unity坐标）
    float2 WorldToScreen(float3 worldPos, float4x4 view, float4x4 proj, float2 screenSize)
    {
        float4 viewPos = mul(view, float4(worldPos, 1.0));
        float4 clipPos = mul(proj, viewPos);
        float3 ndc = clipPos.xyz / clipPos.w;
        
        // Unity的NDC转屏幕坐标（Y轴向上，无需翻转）
        float2 screenPos = float2(
            (ndc.x + 1.0) * 0.5 /* screenSize.x*/,
            (ndc.y + 1.0) * 0.5 /* screenSize.y*/
        );
        return screenPos;
    }

    float SphereFade(float3 worldPosition, float3 center, float visibilityDistance, float fadeWidth)
    {
	    float dist = distance(worldPosition, center);
	    return smoothstep(visibilityDistance, visibilityDistance + fadeWidth, dist);
    }

    float3 clampColorTo01(float3 col)
	{
		float maxChannel = Max3(col.x, col.y, col.z);
		return col * (1 / maxChannel);
	}

    float4 CustomPostProcess(Varyings IN) : SV_Target
    {
        UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

    	half4 outColor = 0;
		half outdepthmask = 1;

    	// float2 uv = ClampAndScaleUVForBilinearPostProcessTexture(IN.texcoord.xy);
    	float2 uv = (IN.texcoord.xy);
        // Note that if HDUtils.DrawFullScreen is not used to render the post process, you don't need to call ClampAndScaleUVForBilinearPostProcessTexture.
        float4 col = SAMPLE_TEXTURE2D_X(_BlitTexture, s_linear_clamp_sampler, uv);
    	float saturation = Max3(col.r, col.g, col.b) - Min3(col.r, col.g, col.b);
    	// col.rgb = clampColorTo01(col.rgb);

    	// return float4(saturation.xxx, 1);
    	// return saturate(col);
    	
		float3 negativeCol = saturate(_NegativeRGBOffset.xyz - col.rgb);
    	float3 hsvCol = RgbToHsv(negativeCol);
    	hsvCol.x += sin(_Time.x * 6 * _NegativeRGBOffset.w);
    	hsvCol.x = fmod(hsvCol.x, 1);
    	negativeCol = HsvToRgb(hsvCol);
		// col.rgb = negativeCol;
    	// return col;

    	
		//sphere space
    	float depth = 1;
		// float depth = SAMPLE_TEXTURE2D_X(_CameraDepthTexture, sampler_CameraDepthTexture, uv).r;	//采样到多Tile的深度图
    	// depth = SAMPLE_TEXTURE2D_X(_DepthTex, sampler_DepthTex, uv).r;		// 0.21左右
		// depth = SampleCameraDepth(uv);
    	#ifndef UNITY_REVERSED_Z
            // Adjust z to match NDC for OpenGL,手机移动平台上面就需要做这样的处理。
            // UNITY_NEAR_CLIP_VALUE：在Direct3d类平台为0，OpenGL类平台为-1.
            depth = lerp(UNITY_NEAR_CLIP_VALUE, 1, depth);
        #endif

    	float mode1 = 1;
    	float mode2 = 0;
    	float mode3 = 0;
    	float mode4 = 0;
    	float mode5 = 0;
    	float mode6 = 1;
    	float2 screenUV = uv;
    	half fadeRange = _NegativeSphereFadeParams.x;
    	half fadeSmooth = fadeRange + _NegativeSphereFadeParams.y;
    	float3 sphereCenterWS = _NegativeSphereParams.xyz;
    	float sphereRadiusWS = _NegativeSphereParams.w;	//值为0.25时可以与sphere的尺寸对上，range为0，smooth为0.001
    	float aspect = _ScreenParams.x / _ScreenParams.y;

    	if (mode1)	//重建世界空间圆形遮罩
    	{
    		// 重新实现：使用类似Unity ComputeScreenPos的方法

    		// 方法：使用深度缓冲重建世界位置来做精确测试
    		// 首先计算每个像素的世界空间位置
    		float depth = SAMPLE_TEXTURE2D_X(_DepthTex, sampler_DepthTex, uv).r;

    		#ifndef UNITY_REVERSED_Z
    			depth = lerp(UNITY_NEAR_CLIP_VALUE, 1, depth);
    		#endif

    		// 重建当前像素的世界空间位置
    		float3 positionWS = ComputeWorldSpacePosition(uv, depth, UNITY_MATRIX_I_VP);

    		// 计算该像素到球心的距离
    		float distToSphereCenter = distance(positionWS, sphereCenterWS) - sphereRadiusWS;

    		// 如果在球体内，应用负片效果
    		float sphereMask = 1.0 - smoothstep(fadeRange, fadeSmooth, distToSphereCenter);
    		col.rgb = lerp(col.rgb, negativeCol.rgb, sphereMask);

    		// 可选：显示球心位置标记
    		float4 centerNDC = mul(UNITY_MATRIX_VP, float4(sphereCenterWS, 1.0));
    		centerNDC.xy /= centerNDC.w;
    		float2 centerSS = centerNDC.xy * 0.5 + 0.5;
    		centerSS.y = 1 - centerSS.y;
    		if (length(uv - centerSS) < 0.01)
    			col.rgb = float3(1, 0, 0);
    	}
		else if (mode2)	//屏幕空间圆形遮罩（自动适配透视变形）
		{
    		// 1. 计算球心的屏幕空间位置
    		float4 centerClip = mul(UNITY_MATRIX_VP, float4(sphereCenterWS, 1.0));
    		float2 centerNDC = centerClip.xy / centerClip.w;

    		if (centerClip.w <= 0)
    			return col;

    		float2 centerSS = centerNDC * 0.5 + 0.5;
    		centerSS.y = 1 - centerSS.y;

    		// 2. 计算球体在屏幕空间的投影半径
    		float3 centerVS = mul(UNITY_MATRIX_V, float4(sphereCenterWS, 1.0)).xyz;

    		float3 sphereTopVS = float3(centerVS.x, centerVS.y + sphereRadiusWS, centerVS.z);
    		float3 sphereBottomVS = float3(centerVS.x, centerVS.y - sphereRadiusWS, centerVS.z);

    		float2 sphereTopNDC = mul(UNITY_MATRIX_P, float4(sphereTopVS, 1.0)).xy / mul(UNITY_MATRIX_P, float4(sphereTopVS, 1.0)).w;
    		float2 sphereBottomNDC = mul(UNITY_MATRIX_P, float4(sphereBottomVS, 1.0)).xy / mul(UNITY_MATRIX_P, float4(sphereBottomVS, 1.0)).w;

    		float radiusSS = abs(sphereTopNDC.y - sphereBottomNDC.y) * 0.5;

    		// 3. 计算从屏幕中心到球心的方向（透视变形方向）
    		float2 screenCenter = float2(0.5, 0.5);
    		float2 toSphere = centerSS - screenCenter;
    		float distFromCenter = length(toSphere);

    		// 4. 计算透视变形系数（离中心越远，变形越大）
    		// 使用球心到相机的距离来计算
    		float distToCamera = -centerVS.z;
    		float tanHalfFov = 1.0 / UNITY_MATRIX_P._m11;

    		// 透视变形因子：距离屏幕中心的归一化距离
    		float2 toSphereNormalized = toSphere / max(screenCenter, float2(0.001, 0.001));
    		float perspectiveFactor = 1.0 + 0.3 * length(toSphereNormalized);  // 可调整0.3来控制变形程度

    		// 5. 计算变形后的椭圆轴
    		// 径向方向（从中心到球心）：被拉伸
    		// 切向方向（垂直于径向）：保持不变
    		float2 radialDir = normalize(toSphere);
    		float2 tangentDir = float2(-radialDir.y, radialDir.x);  // 垂直于径向

    		float aspect = _ScreenParams.x / _ScreenParams.y;

    		// 计算当前像素到球心的向量
    		float2 toPixel = uv - centerSS;
    		toPixel.x *= aspect;

    		// 将向量分解到径向和切向
    		float radialComponent = dot(toPixel, radialDir);
    		float tangentComponent = dot(toPixel, tangentDir);

    		// 应用不同的缩放
    		radialComponent /= perspectiveFactor;  // 径向方向被拉伸，所以需要更小的值才能达到边界

    		// 重新组合并计算距离
    		float2 deformedDist = radialComponent * radialDir + tangentComponent * tangentDir;
    		float distDeformed = length(deformedDist);

    		// 应用遮罩
    		float sphereMaskSS = 1.0 - smoothstep(fadeRange, fadeSmooth, distDeformed / radiusSS);
    		col.rgb = lerp(col.rgb, negativeCol.rgb, sphereMaskSS);

    		// 调试：显示红点标记
    		float2 distToCenter = uv - centerSS;
    		distToCenter.x *= aspect;
    		if (length(distToCenter) < radiusSS * 0.05)
    		{
    			col.rgb = float3(1, 0, 0);
    		}
		}
    	else if (mode3)	//屏幕空间圆形遮罩
		{
    		// 使用ComputeNormalizedDeviceCoordinates方法重新计算

    		// 计算世界坐标到NDC的变换
    		float4 centerClip = mul(UNITY_MATRIX_VP, float4(sphereCenterWS, 1.0));
    		float2 centerNDC = centerClip.xy / centerClip.w;

    		// 检查是否在视锥内
    		if (centerClip.w <= 0)
    			return col;

    		// 转换到屏幕空间
    		float2 centerSS = centerNDC * 0.5 + 0.5;
			centerSS.y = 1 - centerSS.y;

    		// 计算屏幕空间半径 - 使用几何投影法（回退到准确的方法）
    		// 1. 将球心转换到视空间
    		float3 centerVS = mul(UNITY_MATRIX_V, float4(sphereCenterWS, 1.0)).xyz;

    		// 2. 构造球体顶部和底部的视空间点
    		float3 sphereTopVS = float3(centerVS.x, centerVS.y + sphereRadiusWS, centerVS.z);
    		float3 sphereBottomVS = float3(centerVS.x, centerVS.y - sphereRadiusWS, centerVS.z);

    		// 3. 投影到NDC空间
    		float4 sphereTopClip = mul(UNITY_MATRIX_P, float4(sphereTopVS, 1.0));
    		float4 sphereBottomClip = mul(UNITY_MATRIX_P, float4(sphereBottomVS, 1.0));

    		// 4. 透视除法得到NDC坐标
    		float2 sphereTopNDC = sphereTopClip.xy / sphereTopClip.w;
    		float2 sphereBottomNDC = sphereBottomClip.xy / sphereBottomClip.w;

    		// 5. 计算NDC空间的直径和半径
    		float diameterNDC = sphereTopNDC.y - sphereBottomNDC.y;
    		float radiusSS = abs(diameterNDC) * 0.5;

    		// 考虑宽高比
    		float aspect = _ScreenParams.x / _ScreenParams.y;
    		float2 uvAdjusted = uv;
    		uvAdjusted.x *= aspect;
    		float2 centerSSAdjusted = centerSS;
    		centerSSAdjusted.x *= aspect;

    		// 计算遮罩
    		float sphereMaskSS = SphereFadeSS(half2(fadeRange, fadeSmooth), centerSSAdjusted, radiusSS, uvAdjusted);

			// 调试：在球心位置显示一个小红点（大小随透视缩放）
    		float2 distToCenter = uvAdjusted - centerSSAdjusted;
    		float distToCenterLength = length(distToCenter);
    		// 红点大小与圆形遮罩半径成比例，随透视缩放
    		float dotRadius = radiusSS * 0.05;  // 红点半径为圆形遮罩的10%
    		if (distToCenterLength < dotRadius)
    		{
    			// 在红点范围内，显示红色
    			col.rgb = float3(1, 0, 0);
    		}
    		else
    		{
    			// 在红点外，应用圆形遮罩
    			col.rgb = lerp(col.rgb, negativeCol.rgb, sphereMaskSS);
    		}
		}
    	else if (mode4)
    	{
    		float3 camWS = _WorldSpaceCameraPos;

	        float4 clipPos = float4(uv * 2 - 1, -1, 1); // NDC坐标（-1~1）
	        float4 viewPos = mul(UNITY_MATRIX_I_P, clipPos); // 裁剪空间→视口空间
	        viewPos /= viewPos.w; // 透视除法
	        float3 rayWS = mul(UNITY_MATRIX_I_V, float4(viewPos.xyz, 0)).xyz; // 视口空间→世界空间射线方向
	        rayWS = normalize(rayWS);
        
    		// 3. 检测射线与球体的交点
            float t = SphereIntersect(camWS, rayWS, sphereCenterWS, sphereRadiusWS * 10);
    		return float4(t.xxx, 1);
            if (t > 0.0) // 有交点→渲染球体
            {
            	// return float4(1,0,0, 1);

                // 计算球体表面位置和法向量
                float3 posWS = camWS + t * rayWS;
                float3 norWS = normalize(posWS - sphereCenterWS);

                // 球体基础着色（保留原漫反射逻辑，适配Unity参数）
                float3 sphereCol = col.rgb;
                sphereCol *= 0.6 + 0.4 * norWS.y; // 法向量y分量影响亮度
                
                // 距离衰减（和原代码一致）
                sphereCol *= exp(-0.05 * t);

                // 叠加球体颜色（替换/混合可选，这里用球体颜色覆盖原像素）
                col.rgb = negativeCol.rgb;
            }
    	}
    	else if (mode5)	//边界框投影法：通过投影球体边界计算精确的椭圆遮罩
    	{
    		// 算法原理：
    		// 1. 在视空间中构造球体的6个边界点（上下左右前后）
    		// 2. 将这些点投影到屏幕空间
    		// 3. 使用投影后的边界计算椭圆参数
    		// 4. 这样可以精确匹配球体的透视变形

    		// 转换球心到视空间
    		float3 centerVS = mul(UNITY_MATRIX_V, float4(sphereCenterWS, 1.0)).xyz;

    		// 检查球是否在相机前方
    		if (centerVS.z < 0)
    		{
    			// 在视空间中构造球体的边界点
    			float3 spherePointsVS[6];
    			spherePointsVS[0] = centerVS + float3(0, sphereRadiusWS, 0);  // 上
    			spherePointsVS[1] = centerVS + float3(0, -sphereRadiusWS, 0); // 下
    			spherePointsVS[2] = centerVS + float3(sphereRadiusWS, 0, 0);  // 右
    			spherePointsVS[3] = centerVS + float3(-sphereRadiusWS, 0, 0); // 左
    			spherePointsVS[4] = centerVS + float3(0, 0, sphereRadiusWS);  // 前
    			spherePointsVS[5] = centerVS + float3(0, 0, -sphereRadiusWS); // 后

    			// 投影到屏幕空间并计算边界
    			float2 minSS = float2(1, 1);
    			float2 maxSS = float2(0, 0);

    			for (int i = 0; i < 6; i++)
    			{
    				float4 clipPos = mul(UNITY_MATRIX_P, float4(spherePointsVS[i], 1.0));
    				if (clipPos.w > 0)
    				{
    					float2 ndc = clipPos.xy / clipPos.w;
    					float2 screenPos = ndc * 0.5 + 0.5;
    					screenPos.y = 1 - screenPos.y;  // Y轴翻转
    					minSS = min(minSS, screenPos);
    					maxSS = max(maxSS, screenPos);
    				}
    			}

    			// 计算椭圆中心和半轴
    			float2 ellipseCenter = (minSS + maxSS) * 0.5;
    			float2 ellipseRadii = (maxSS - minSS) * 0.5;

    			// 考虑宽高比
    			float aspect = _ScreenParams.x / _ScreenParams.y;
    			float2 uvAdjusted = uv;
    			uvAdjusted.x *= aspect;
    			float2 centerAdjusted = ellipseCenter;
    			centerAdjusted.x *= aspect;
    			float2 radiiAdjusted = ellipseRadii;
    			radiiAdjusted.x *= aspect;

    			// 计算当前像素到椭圆中心的距离
    			float2 toPixel = uvAdjusted - centerAdjusted;
    			float normalizedDist = length(toPixel / radiiAdjusted);

    			// 应用椭圆遮罩
    			float sphereMask = 1.0 - smoothstep(fadeRange, fadeSmooth, normalizedDist - 1.0);
    			col.rgb = lerp(col.rgb, negativeCol.rgb, sphereMask);

    			// 调试：显示球心红点
    			float2 distToCenter = uvAdjusted - centerAdjusted;
    			if (length(distToCenter) < 0.02)
    			{
    				col.rgb = float3(1, 0, 0);
    			}
    		}
    	}
    	else if (mode6)	//射线-球相交法（改进版）：不考虑深度，只看射线是否穿过球体的3D投影
    	{
    		// 核心思想：
    		// 对于每个像素的射线，计算它到球心的垂直距离
    		// 如果距离小于半径，说明射线穿过球体，显示负片效果
    		// 这就是真正的3D球体在2D屏幕上的投影

    		float3 rayDirWS = IN.rayWS;

    		// 相机到球心的向量
    		float3 camToSphere = sphereCenterWS - _WorldSpaceCameraPos;

    		// 计算射线上距离球心最近的点
    		float t = dot(camToSphere, rayDirWS);

    		// 最近点的坐标
    		float3 closestPoint = _WorldSpaceCameraPos + t * rayDirWS;

    		// 最近点到球心的距离（垂直距离）
    		float distToSphere = distance(closestPoint, sphereCenterWS);

    		// 判断：如果垂直距离小于半径，说明射线穿过球体
    		if (distToSphere < sphereRadiusWS)
    		{
    			// 射线穿过球体，直接显示负片效果
    			col.rgb = negativeCol.rgb;
    		}
    		else
    		{
    			// 射线不穿过球体，使用边缘平滑过渡
    			float distFromSurface = distToSphere - sphereRadiusWS;

    			// 需要将距离转换到屏幕空间尺度
    			// 根据球体的深度来缩放距离
    			float depthToSphere = length(camToSphere);
    			float screenScale = 1.0 / (depthToSphere * 0.5);  // 调整系数

    			float sphereMask = 1.0 - smoothstep(fadeRange * screenScale, fadeSmooth * screenScale, distFromSurface);
    			col.rgb = lerp(col.rgb, negativeCol.rgb, sphereMask);
    		}

    		// 调试：显示球心红点
    		float4 centerClip = mul(UNITY_MATRIX_VP, float4(sphereCenterWS, 1.0));
    		if (centerClip.w > 0)
    		{
    			float2 centerSS = centerClip.xy / centerClip.w * 0.5 + 0.5;
    			centerSS.y = 1 - centerSS.y;
    			float aspect = _ScreenParams.x / _ScreenParams.y;
    			float2 distToCenter = uv - centerSS;
    			distToCenter.x *= aspect;
    			if (length(distToCenter) < 0.02)
    			{
    				col.rgb = float3(1, 0, 0);
    			}
    		}
    	}
    	// else if (mode4)
	    // {
    	// 	// 2. 计算相机到球体的真实距离
     //        float3 camWorldPos = _WorldSpaceCameraPos;
     //        float actualCamDist = length(camWorldPos - sphereCenterWS);
    	// 	// 3. 计算球体中心的屏幕坐标
     //        float2 screenPos = WorldToScreen(sphereCenterWS, UNITY_MATRIX_V, UNITY_MATRIX_P, _ScreenParams.xy);
    	// 	// 4. 计算球体真实投影半径
     //        float screenRadius = GetSphereScreenRadius(sphereRadiusWS, actualCamDist, UNITY_MATRIX_P, _ScreenParams.y);
     //
    	// 	float dist = distance(uv, screenPos) - screenRadius;
    	// 	float sphereMaskSS = SphereFadeSS(half2(fadeRange, fadeSmooth), screenPos, screenRadius, screenUV);
    	// 	col.rgb = lerp(col.rgb, negativeCol.rgb, sphereMaskSS);
    	// 	// return float4(sphereMaskSS.xxx, 1);
     //        // if (dist < screenRadius)
     //        // {
     //        //     col.rgb = negativeCol.rgb;
     //        // 	return float4(negativeCol.rgb, 1);
     //        // }
	    // }
    	
    	// float sphereMaskWS = SphereFadeWS(half2(fadeRange, fadeSmooth), _NegativeSphereParams.xyz, _NegativeSphereParams.w, IN.texcoord, uv);
    	// float sphereMaskVS = SphereFadeVS(half2(fadeRange, fadeSmooth), _NegativeSphereParams.xyz, _NegativeSphereParams.w * 5, IN.texcoord, uv);

    	// col.rgb = float3(centerPosSS, 0);
    	// col.rgb = float3(frac(IN.texcoord.xy * 5), 0);
    	// col.rgb = float3(length(uv - centerPosSS).xx, 0);
    	
  //   	#if _USEDEPTHMASK
		// 	half depthmask = SAMPLE_TEXTURE2D_X(_CameraDepthTexture, sampler_CameraDepthTexture, uv);
		// 	outdepthmask = LinearEyeDepth(depthmask, _ZBufferParams);
		// #endif
  //
  //   	float2 CenterUV = IN.texcoord.xy - _BlurCenter ;
  //       float Polar = length( CenterUV ) * _RadialStretchRange;
  //       float ZoomUV = saturate(1.0 - _RadialStretching * (1.0 - Polar));
  //       float2 OffsetUV = (0.5f / ZoomUV).xx;
  //
		// for (float j = 0.0; j < 1.0; j += 1.0f / LOOPCOUNT) 
		// {
		// 	float alpha = j * blurStrength*0.2;
		// 	
		// 	float tiling = 0;
		// 	float2 offset = float2(0, 0);
		// 	float2 uv = float2(0, 0);
		// 	tiling = 1 - alpha;
		// 	offset = _BlurCenter * alpha;
		// 	uv = IN.texcoord * tiling + offset;
  //           uv = lerp(uv / ZoomUV + 0.5 - OffsetUV, uv, Polar.x);
		// 	uv = ClampAndScaleUVForBilinearPostProcessTexture(uv);
		// 	
		// 	#if _USEDEPTHMASK
		// 	half tempdepthmask = SAMPLE_TEXTURE2D_X(_CameraDepthTexture, sampler_CameraDepthTexture, uv);
		// 	float tempdepth = LinearEyeDepth(tempdepthmask, _ZBufferParams);
		// 	offset = lerp(0, offset, step(_ClipDepth, tempdepth));
		// 	tiling = lerp(1, 1 - alpha, step(_ClipDepth, tempdepth));
		// 	uv = IN.texcoord * tiling + offset;
		// 	#endif
		// 	
		// 	outColor += SAMPLE_TEXTURE2D_X(_BlitTexture, s_linear_clamp_sampler, uv);
		// 	
		// }
		// outColor *= 1.0f / LOOPCOUNT;
  //
  //   	float blurRadius = _RadialBlurParameter.y + 0.5;
		// float definition = saturate( _RadialBlurParameter.a);
		// half alpha  = length((IN.texcoord - _BlurCenter.xy)) * 2;
  //
  //   	#if _RADIALINVERT
		// alpha = saturate(4 * blurRadius - alpha);
		// #else
		// alpha = 1 - saturate(4 * (1 - blurRadius) - alpha);
		// #endif
		// #if _USEDEPTHMASK
		// col.rgb = lerp(col.rgb, outColor.rgb, alpha * definition * step(_ClipDepth, outdepthmask));
		// #else
		// col.rgb = lerp(col.rgb, outColor.rgb, alpha * definition);
		// #endif

        return col;
    }

    ENDHLSL

    SubShader
    {
        Tags{ "RenderPipeline" = "URPRenderPipeline" }
        Pass
        {
            Name "NegativeColor"

            ZWrite Off
            ZTest Always
            Blend Off
            Cull Off

            HLSLPROGRAM
                #pragma fragment CustomPostProcess
                #pragma vertex Vert
            ENDHLSL
        }
    }
    Fallback Off
}
