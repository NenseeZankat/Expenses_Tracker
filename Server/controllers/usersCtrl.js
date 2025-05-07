const asyncHandler = require("express-async-handler");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const User = require("../model/User");
const multer = require("multer");
//!User Registration


const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/"); // ✅ Store in 'uploads' folder
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`); // ✅ Unique filename
  },
});

const upload = multer({ storage:storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // Limit to 5MB
 });
const usersController = {
  //!Register
  register: asyncHandler(async (req, res) => {
    const { username, email, password} = req.body;
    //!Validate
    if (!username || !email || !password) {
      throw new Error("Please all fields are required");
    }
    //!Check if user already exists
    const userExists = await User.findOne({ email });
    if (userExists) {
      throw new Error("User already exists");
    }
    //!Hash the user password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);
    //! Create the user and save into db
    const userCreated = await User.create({
      email,
      username,
      password: hashedPassword,
      //role:"individual",
    });
    //! Send the response

    res.json({
      username: userCreated.username,
      email: userCreated.email,
      id: userCreated._id,
      //role:userCreated.role,
    });
  }),
  //! Find user by email (✅ Added here)
  findUserByEmail: asyncHandler(async (req, res) => {
    console.log(req.params);
    const email = req.params.email;

    if (!email) {
        return res.status(400).json({ message: "Email is required" });
    }

    const user = await User.findOne({ email: email.toLowerCase() });

    if (!user) {
        return res.status(404).json({ message: "User not found" });
    }

    // Return only the ID
    res.status(200).json({ id: user._id });
}),

  
  //!Login
  login: asyncHandler(async (req, res) => {
    //! Get the user data
    const { email, password } = req.body;
    //!check if email is valid
    const user = await User.findOne({ email });
    if (!user) {
      throw new Error("Invalid login credentials");
    }
    // if (user.role !== role) {
    //   return res
    //     .status(403)
    //     .json({ message: `User does not have the '${role}' role` });
    // }
    //! Compare the user password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      throw new Error("Invalid login credentials");
    }
    //! Generate a token
    const token = jwt.sign({ id: user._id }, "masynctechKey", {
      expiresIn: "30d",
    });
    //!Send the response
    res.json({
      message: "Login Success",
      token,
      id: user._id,
      email: user.email,
      username: user.username,
      familyId:user.familyId,
      //role:user.role,
      profilePic: user.profilePic, // ✅ Return profile picture
    });
  }),

  //!profile
  profile: asyncHandler(async (req, res) => {
    //! Find the user
    console.log(req.user);
    const user = await User.findById(req.user);
    if (!user) {
      throw new Error("User not found");
    }
    //!Send the response
    res.json({ username: user.username, email: user.email, familyId:user.familyId});//,role:user.role });
  }),
  //! Change Password
  changeUserPassword: asyncHandler(async (req, res) => {
    const { newPassword } = req.body;
    const user = await User.findById(req.user);
    if (!user) {
      throw new Error("User not found");
    }

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(newPassword, salt);
    user.password = hashedPassword;

    await user.save({ validateBeforeSave: false });

    res.json({ message: "Password Changed successfully" });
  }),

  //! Update Username
  updateUserProfile: asyncHandler(async (req, res) => {
    const { username } = req.body;

    const updatedUser = await User.findByIdAndUpdate(
      req.user,
      { username },
      { new: true }
    );

    if (!updatedUser) {
      throw new Error("User not found");
    }

    res.json({ message: "User profile updated successfully", data: updatedUser });
  }),

  uploadProfilePic: asyncHandler(async (req, res) => {
    console.log("file",req.file);
    if (!req.file) {
      return res.status(400).json({ message: "No file uploaded" });
    }
  
    const profilePic = `/uploads/${req.file.filename}`; // ✅ Save file path
    console.log(req.user);  // User ID from authenticated request

    const updatedUser = await User.findByIdAndUpdate(
      req.user,  // User ID from authenticated request
      { profilePic },  // Update only profilePic field
      { new: true }  // Return updated document
    );
  
    if (!updatedUser) {
      throw new Error("User not found");
    }
  
    res.json({ message: "Profile picture updated successfully", profilePic });
  }),
};





module.exports = usersController;
