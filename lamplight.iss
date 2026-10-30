; Inno Setup script: builds LamplightSetup.exe from the PyInstaller output in dist\Lamplight
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
[Setup]
AppId={{6D1F7B52-3C0E-4F6A-9B1D-4C7A2E5B9F10}
AppName=Lamplight
AppVersion={#AppVersion}
AppPublisher=Ivan De Leon
DefaultDirName={localappdata}\Programs\Lamplight
DefaultGroupName=Lamplight
PrivilegesRequired=lowest
OutputDir=installer
OutputBaseFilename=LamplightSetup
SetupIconFile=assets\lamplight.ico
UninstallDisplayIcon={app}\Lamplight.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "dist\Lamplight\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Lamplight"; Filename: "{app}\Lamplight.exe"
Name: "{autodesktop}\Lamplight"; Filename: "{app}\Lamplight.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Put a Lamplight icon on my desktop"; Flags: checkedonce

[Run]
Filename: "{app}\Lamplight.exe"; Description: "Open Lamplight now"; Flags: nowait postinstall skipifsilent
