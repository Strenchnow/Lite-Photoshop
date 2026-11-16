; LitePhotoEditor — Inno Setup 6 Script
; Build with: python build_installer.py  (or ISCC.exe installer\setup.iss)

#define MyAppName "LitePhotoEditor"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "LitePhotoEditor"
#define MyAppExeName "LitePhotoEditor.exe"

[Setup]
AppId={{8E7B9C1A-4F2D-4C6B-9E12-7A3D5F8C2B10}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=LitePhotoEditor_Setup
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
WizardStyle=modern
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
russian.StartMenuIconTask=Создать ярлык в меню «Пуск»
english.StartMenuIconTask=Create a Start Menu shortcut
russian.ContextMenuTask=Добавить в контекстное меню Windows «Открыть в LitePhotoEditor»
english.ContextMenuTask=Add "Open with LitePhotoEditor" to Explorer context menu
russian.ContextMenuVerb=Открыть в LitePhotoEditor
english.ContextMenuVerb=Open with LitePhotoEditor

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "startmenuicon"; Description: "{cm:StartMenuIconTask}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "contextmenu"; Description: "{cm:ContextMenuTask}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\LitePhotoEditor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: startmenuicon
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; Optional Explorer "Open with LitePhotoEditor" context menu for supported image formats
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\LitePhotoEditor"; ValueType: string; ValueName: ""; ValueData: "{cm:ContextMenuVerb}"; Tasks: contextmenu; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\LitePhotoEditor"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"",0"; Tasks: contextmenu
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\LitePhotoEditor\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: contextmenu

Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\LitePhotoEditor"; ValueType: string; ValueName: ""; ValueData: "{cm:ContextMenuVerb}"; Tasks: contextmenu; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\LitePhotoEditor"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"",0"; Tasks: contextmenu
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\LitePhotoEditor\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: contextmenu

Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\LitePhotoEditor"; ValueType: string; ValueName: ""; ValueData: "{cm:ContextMenuVerb}"; Tasks: contextmenu; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\LitePhotoEditor"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"",0"; Tasks: contextmenu
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\LitePhotoEditor\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: contextmenu

Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\LitePhotoEditor"; ValueType: string; ValueName: ""; ValueData: "{cm:ContextMenuVerb}"; Tasks: contextmenu; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\LitePhotoEditor"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"",0"; Tasks: contextmenu
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\LitePhotoEditor\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: contextmenu

Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\LitePhotoEditor"; ValueType: string; ValueName: ""; ValueData: "{cm:ContextMenuVerb}"; Tasks: contextmenu; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\LitePhotoEditor"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"",0"; Tasks: contextmenu
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\LitePhotoEditor\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: contextmenu

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
