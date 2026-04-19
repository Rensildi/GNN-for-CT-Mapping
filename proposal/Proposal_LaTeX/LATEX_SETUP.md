# GNN-for-CT-Mapping

# LaTeX Setup in VS Code (Windows and Ubuntu)

This guide explains how to set up **LaTeX in Visual Studio Code** for both **Windows** and **Ubuntu Linux** users.

## What you need

No matter which operating system you use, you need these two things:

1. **Visual Studio Code**
2. **LaTeX Workshop** extension in VS Code

After that, each operating system also needs a **LaTeX distribution** installed on the computer.

---

# Windows Setup

For Windows, this guide uses:

- **MiKTeX**
- **Strawberry Perl**
- **LaTeX Workshop**

## Step 1: Install Visual Studio Code

Download and install Visual Studio Code from the official website.

## Step 2: Install the LaTeX Workshop extension

1. Open **VS Code**.
2. Click the **Extensions** icon on the left sidebar.
3. Search for **LaTeX Workshop**.
4. Install the extension by **James-Yu**.

## Step 3: Install MiKTeX

1. Go to the official MiKTeX download page.
2. Download the **Basic MiKTeX Installer** for Windows.
3. Run the installer.
4. Keep the default settings unless you have a specific reason to change them.
5. When installation finishes, open **MiKTeX Console** once and allow it to complete any first-time setup.

### Recommended MiKTeX settings

Inside **MiKTeX Console**:

- Update packages if updates are available.
- Turn on **install missing packages on-the-fly** if it is not already enabled.

This helps MiKTeX automatically install packages that your `.tex` file needs.

## Step 4: Install Strawberry Perl

1. Go to the official Strawberry Perl website.
2. Download the **64-bit MSI** installer.
3. Run the installer.
4. Restart VS Code after installation.

> Why is Strawberry Perl needed?
> 
> LaTeX Workshop commonly uses `latexmk` for building LaTeX projects, and `latexmk` requires **Perl**.

## Step 5: Check that the tools are available

Open **Command Prompt** or the **VS Code terminal** and run:

```bash
pdflatex --version
latexmk --version
perl --version
```

If these commands return version information, the setup is working.

## Step 6: Build the PDF in VS Code

1. Open the folder that contains `test.tex` in VS Code.
2. Open `main.tex`.
3. Build the document using one of these methods:
   - Click the **Build LaTeX project** button from LaTeX Workshop, or
   - Open the Command Palette and run the LaTeX build command.
4. Open the PDF preview in VS Code.

If everything is set up correctly, a PDF will be generated.


# Ubuntu Setup

For **Ubuntu**, the easiest and most reliable setup is usually:

- **TeX Live**
- **LaTeX Workshop**

Although **MiKTeX is available on selected Linux distributions**, **TeX Live is usually the simpler and more recommended option on Linux**.

## Step 1: Install Visual Studio Code

Install Visual Studio Code on Ubuntu.

## Step 2: Install the LaTeX Workshop extension

1. Open **VS Code**.
2. Go to **Extensions**.
3. Search for **LaTeX Workshop**.
4. Install it.

## Step 3: Install TeX Live

Open a terminal and run:

```bash
sudo apt update
sudo apt install texlive-full
```

This is the easiest option because it installs a complete LaTeX environment and also pulls in tools commonly needed for building documents.

> Note:
> `texlive-full` is large. It takes more disk space, but it avoids many missing-package issues.

## Step 4: Check that the tools are available

Run these commands:

```bash
pdflatex --version
latexmk --version
```

If both commands return version information, the installation is ready.

## Step 5: Build the PDF in VS Code

1. Open the project folder in **VS Code**.
2. Open `test.tex`.
3. Run the LaTeX build command from LaTeX Workshop.
4. Open the generated PDF preview.

---

# Ubuntu Alternative: MiKTeX instead of TeX Live

If you specifically want to use **MiKTeX on Ubuntu**, it is supported only for **selected Linux distributions**.

In that case:

1. Open the official MiKTeX Linux installation page.
2. Follow the Linux-specific instructions for your Ubuntu version.
3. Make sure the LaTeX binaries are available in your system `PATH`.
4. Verify the installation using:

```bash
pdflatex --version
latexmk --version
```

Because Linux support can vary by distribution and version, it is safer to follow the official MiKTeX instructions directly instead of copying old commands from random websites.

---

# Troubleshooting

## Problem: `latexmk` not found

This usually means the LaTeX build tool is not installed or not in your system `PATH`.

- On **Windows**, make sure MiKTeX installed correctly and restart VS Code.
- On **Ubuntu**, make sure TeX Live is installed properly.

## Problem: `perl` not found on Windows

This usually means **Strawberry Perl** is missing or not available in the terminal session yet.

- Install Strawberry Perl.
- Restart VS Code.
- Try again.

## Problem: PDF does not build in VS Code

Check these:

- The file must be saved as `.tex`
- The main file should contain a valid LaTeX document structure
- The LaTeX tools must be available in the terminal
- Restart VS Code after installing MiKTeX or Perl

## Problem: Missing package errors

- On **MiKTeX**, allow automatic package installation.
- On **Ubuntu with TeX Live**, using `texlive-full` avoids most missing-package errors.

---

# Recommended setup summary

## Windows

Use:

- VS Code
- LaTeX Workshop
- MiKTeX
- Strawberry Perl

## Ubuntu

Recommended:

- VS Code
- LaTeX Workshop
- TeX Live (`texlive-full`)

Alternative:

- VS Code
- LaTeX Workshop
- MiKTeX for Linux (only if you specifically want MiKTeX)

---

# References

- LaTeX Workshop (VS Code Marketplace): https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop
- LaTeX Workshop installation wiki: https://github.com/James-Yu/LaTeX-Workshop/wiki/Install
- MiKTeX download page: https://miktex.org/download
- MiKTeX Linux installation page: https://miktex.org/howto/install-miktex-unx
- Strawberry Perl: https://strawberryperl.com/
- TeX Live: https://www.tug.org/texlive/
