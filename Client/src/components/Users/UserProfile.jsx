import React, { useEffect, useState } from "react";
import { useSelector, useDispatch } from "react-redux";

import { FaUserCircle, FaEnvelope, FaLock,FaCamera,FaUserShield, } from "react-icons/fa";
import { useFormik } from "formik";
import { useMutation, useQuery } from "@tanstack/react-query";
import UpdatePassword from "./UpdatePassword";
import { updateProfilePicAPI,updateProfileAPI, getUserPreferenceAPI, updateUserPreferenceAPI,getUserDetailsAPI, } from "../../services/users/userService";
import AlertMessage from "../Alert/AlertMessage";
import { setProfilePic } from "../../redux/slice/userSlice";
import { updateUser } from "../../redux/slice/authSlice";
import { useLocation } from "react-router-dom";


const UserProfile = ({ userId }) => {
  const dispatch = useDispatch();
  const location = useLocation();

  const [username, setUsername] = useState("User");
  const [profilePic, setProfilePicState] = useState("/default-profile.png");
  const [showFullImage, setShowFullImage] = useState(false);

  const token = useSelector((state) => state.auth.token);
  const userFromRedux = useSelector((state) => state.auth?.user);
  const profilePicFromRedux = useSelector((state) => state.user.profilePic);

  // ✅ Load user from localStorage and fallback to API if needed
  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    const storedToken = localStorage.getItem("token");

    if (storedToken && !userFromRedux?.username) {
      //console.log("HIIIIII");
      getUserDetailsAPI(storedToken)
        .then((res) => {
          dispatch(updateUser(res.user));
          if (res.user?.profilePic) {
            const imgUrl = `https://backend-johh.onrender.com${res.user.profilePic}`;
            dispatch(setProfilePic(imgUrl));
            setProfilePicState(imgUrl);
            localStorage.setItem("profilePic", imgUrl);
          }
          localStorage.setItem("user", JSON.stringify(res.user));
          //console.log("HIIIIII");
        })
        .catch((err) => console.error("Failed to fetch user from API", err));
    } else if (storedUser) {
      try {
        const parsed = JSON.parse(storedUser);
        dispatch(updateUser(parsed));
        if (parsed?.username) setUsername(parsed.username);
      } catch (error) {
        console.error("Error parsing user data from localStorage:", error);
      }
    }
    
  }, [dispatch]);

  // ✅ Keep local username in sync with Redux (and on route change)
  useEffect(() => {
    if (userFromRedux?.username) {
      setUsername(userFromRedux.username);
    }
  }, [userFromRedux?.username, location.pathname]);

  // ✅ Load user preference
  const { data: userPreference } = useQuery({
    queryKey: ["user-preference", userId],
    queryFn: () => getUserPreferenceAPI(userId),
    enabled: !!userId,
  });

  // ✅ Set profile picture
  useEffect(() => {
    const storedProfilePic = localStorage.getItem("profilePic");
    
    if (storedProfilePic) {
      setProfilePicState(storedProfilePic);
    } else if (userPreference?.profilePic) {
      const imageUrl = `https://backend-johh.onrender.com${userPreference.profilePic}`;
      setProfilePicState(imageUrl);
      localStorage.setItem("profilePic", imageUrl);
    }
    }, [userPreference]);

  // ✅ Username update mutation
  const updateUsernameMutation = useMutation({

    mutationFn: (values) =>
      updateProfileAPI({ username: values.username }, token),
    onSuccess: (response) => {
      const updatedUsername = response?.data.username;
      console.log(updatedUsername);
      if (updatedUsername) {
        
        localStorage.setItem("user", JSON.stringify(response.updatedUser));
        setUsername(updatedUsername);
        console.log("✅ Username updated to:", updatedUsername);
        dispatch(updateUser(response.data.username));
      }
      
    },
    onError: (error) => {
      console.error("Error updating username:", error);
    },
  }
  
);

  const formikUsername = useFormik({
    initialValues: {
      username: username || "",
    },
    enableReinitialize: true,
    onSubmit: (values) => {
      console.log(values.username);
      updateUsernameMutation.mutate(values);
    },
  });

  // ✅ Profile picture update
  const updateProfilePicMutation = useMutation({
    mutationFn: (formData) => updateProfilePicAPI(formData, token),
    onSuccess: (response) => {
      if (response?.profilePic) {
        const imageUrl = `https://backend-johh.onrender.com${response.profilePic}`;
        dispatch(setProfilePic(imageUrl));
        setProfilePicState(imageUrl);
        localStorage.setItem("profilePic", imageUrl);
      }
    },
    onError: (error) => {
      console.error("Error updating profile picture:", error);
    },
  });

  const handleProfilePicChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("profilePic", file);
    await updateProfilePicMutation.mutateAsync(formData);
  };

  return (
    <>
      <div className="max-w-4xl mx-auto my-10 p-8 bg-white rounded-lg shadow-md relative">
        <h1 className="mb-2 text-2xl text-center font-extrabold">Welcome</h1>
        <h3 className="text-xl font-semibold text-gray-800 mb-4">Profile</h3>

        {/* Alerts */}
        {updateUsernameMutation.isLoading && <AlertMessage type="loading" message="Saving changes..." />}
        {updateUsernameMutation.isError && <AlertMessage type="error" message="Failed to update username." />}
        {updateUsernameMutation.isSuccess && <AlertMessage type="success" message="Username updated successfully!" />}

        {updateProfilePicMutation.isLoading && <AlertMessage type="loading" message="Updating picture..." />}
        {updateProfilePicMutation.isError && <AlertMessage type="error" message="Error updating profile picture" />}
        {updateProfilePicMutation.isSuccess && <AlertMessage type="success" message="Profile picture updated successfully" />}

        {/* Profile Picture */}
        <div className="flex flex-col items-center mb-6 relative">
          <img
            src={profilePic}
            alt="Profile"
            className="w-24 h-24 rounded-full shadow-md object-cover cursor-pointer"
            onClick={() => setShowFullImage(true)}
          />
          <label className="mt-2 cursor-pointer flex items-center space-x-2 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
            <FaCamera />
            <span>{profilePic === "/default-profile.png" ? "Upload Profile Picture" : "Update Profile Picture"}</span>
            <input type="file" className="hidden" accept="image/*" onChange={handleProfilePicChange} />
          </label>
        </div>

        {/* Fullscreen Image Modal */}
        {showFullImage && (
          <div
            onClick={() => setShowFullImage(false)}
            className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50"
          >
            <img
              src={profilePic}
              alt="Full Profile"
              className="max-w-full max-h-full rounded-lg shadow-lg"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        )}

        {/* Email */}
        <div className="flex items-center space-x-4 mb-4">
          <FaEnvelope className="text-2xl text-gray-400" />
          <div className="flex-1">
            <label className="text-sm font-medium text-gray-700">Email</label>
            <div className="mt-1 w-full border border-gray-300 rounded-md shadow-sm py-2 px-4 bg-gray-100 text-gray-700">
              {userFromRedux?.email || "Not available"}
            </div>
          </div>
        </div>

        {/* Role */}
        {userFromRedux?.role && (
          <div className="flex items-center space-x-4 mb-6">
            <FaUserShield className="text-2xl text-gray-400" />
            <div className="flex-1">
              <label className="text-sm font-medium text-gray-700">Role</label>
              <div className="mt-1 w-full border border-gray-300 rounded-md shadow-sm py-2 px-4 bg-gray-100 text-gray-700">
                {userFromRedux.role}
              </div>
            </div>
          </div>
        )}

        {/* Username Form */}
        <form onSubmit={formikUsername.handleSubmit} className="space-y-6">
          <div className="flex items-center space-x-4">
            <FaUserCircle className="text-3xl text-gray-400" />
            <div className="flex-1">
              <label htmlFor="username" className="text-sm font-medium text-gray-700">
                Username
              </label>
              <input
                {...formikUsername.getFieldProps("username")}
                type="text"
                id="username"
                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-4 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                placeholder="Your username"
              />
            </div>
          </div>

          <div className="flex justify-end mt-6">
            <button
              type="submit"
              disabled={updateUsernameMutation.isLoading}
              className={`${
                updateUsernameMutation.isLoading
                  ? "bg-blue-300"
                  : "bg-blue-500 hover:bg-blue-700"
              } text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline`}
            >
              {updateUsernameMutation.isLoading ? "Saving..." : "Save Changes"}
            </button>
          </div>
        </form>
      </div>

      {/* Change Password */}
      <div className="max-w-4xl mx-auto my-10 p-8 bg-white rounded-lg shadow-md">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">Change Password</h3>
        <UpdatePassword />
      </div>
    </>
  );
};

export default UserProfile;


 

  

 