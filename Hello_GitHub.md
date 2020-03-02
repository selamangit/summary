###Hello GitHub
**学习GitHub的意义**
GitHub是一个面向开源及私有软件的托管平台，在上面有来自世界各地的程序员发布的开源项目，这是一个很好的学习平台，而且它满足团队协作的优点，能让多人能够同时对一个项目进行编程。作为一个分布式的版本控制系统，它能让各位开发者在GItHub的库上克隆到本地库上进行编码，也可以通过Git上传到服务器上，适合分布开发。
**使用GitHub的方法**
* 首先你需要创建GitHub平台的账号，完善个人资料。
* 其次为了让你的项目有地方存放，你要创建一个仓库（Repository）
![](https://guides.github.com/activities/hello-world/create-new-repo.png)
为了让团队成员和其他人能够理解这个项目的内容，创建的仓库里最好有一个ReadMe.md的说明文档，点击Initialize this repository with a README会自动创建在仓库中。
* **fork**是一个将别人的开源项目添加到自己仓库的一个按钮，fork后的项目可以点击**Clone or download**复制链接通过git clone将文件到本地文件中。
* 为了体现分布开发的特点仓库中还包含了一种叫**分支**（branch）的东西，默认情况下，仓库中会有一个叫做master的主分支，但是可以在主分支中创建分支，团队成员根据分工拉取不同功能的分支进行不同的功能开发，这样就可以隔离每个人的工作，当每个人的分支都完成后，可以向主分支master发起pull Request，在确保这些分支的功能对项目是有效的时候可以将这些分支合并入主分支master中。在仓库的code选项卡中的Branch下拉框里命名新分支的名字就可以创建新的分支
* **Pull Request**是将个人代码提交到团队代码的过程，在GitHub上也可以进行Pull Request的操作，在仓库的界面上就有Pull Request的按钮，点击后进入Compare Change的界面来比较代码。
**GitHub issue**在仓库的选项卡中的按钮在这里可以看到代码报错信息，这个选项还可以做为项目所希望达成的目标的地方，相当于给团队的任务，当某个Pull Request解决了这个问题时这个issue就会消失。
********
在GitHub平台上可以点击Explore去发现其他人的开源项目，通过readme文档的介绍来选择感兴趣的项目，还可以点击你喜欢的人的主页上的watch来追踪他/她的最新动态。
***********
###Hello Git
GitHub只支持Git作为唯一的版本库格式进行托管，通过Git Bash也可以与GitHub上的远程仓库进行连接，它可以将GitHub上的项目取出到本地文件中的一个软件，它与远程仓库通过ssh协议进行连接
![](https://img-blog.csdn.net/20170422114557808?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvanRyYWN5ZHk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
* 提交文件git commit -m“文件名”暂存区中的文件到仓库区中，若想上传到GitHub上还需git push将代码推送到远程仓库中
**常用的命令**
mkdir 在当前目录中创建文件夹
git init 选取当前文件夹作为本地仓库
git clone + 远程仓库的链接  这样就可以将GitHub上的项目拿到本地仓库中进行编码
git status 查看当前的git仓库状态
git add[file1][file2] 添加指定文件到暂存区
git add[dir]添加指定目录到暂存区，包括子目录
git add 添加当前目录的所有文件到暂存区
git rm[file1][file2]停止追踪指定文件，但该文件会保留在工作区（本地文件），取消被跟踪的文件将不会被push到GitHub上去
git commit -m[注释]将暂存区的文件提交到仓库区中
git commit[file1][file2] -m[注释]将暂存区的指定文件提交到仓库区
git commit -a 提交工作区自上一次commit之后的变化，直接到仓库区
git remove -v 查看远程仓库的路径
git remote add origin + 远程仓库的地址 （添加远程仓库）
git remote rm + 远程仓库的地址 删除远程仓库 （删除远程仓库）
**获取ssh密码与github建立联系**
* 先在Git Bash通过git config --global配置全局用户名和邮箱，接着输入ssh-keygen -t rsa -C（大写）“邮箱”来获取密码，获取密码后进入GitHub里将密码粘贴进去即可。