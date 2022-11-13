# Activity

## 注册

所有activity都要在`AndroidManifest.xml`中注册才能生效

注册声明在`<application>`标签内

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="com.example.activitytest">
<application
android:allowBackup="true" android:icon="@mipmap/ic_launcher" android:label="@string/app_name" android:roundIcon="@mipmap/ic_launcher_round" android:supportsRtl="true" android:theme="@style/AppTheme"> <activity android:name=".FirstActivity"> </activity> </application>
</manifest>
```
