package com.xiaoyu.springbootmybatisprac.service.impl;

import com.xiaoyu.springbootmybatisprac.mapper.UserMapper;
import com.xiaoyu.springbootmybatisprac.pojo.User;
import com.xiaoyu.springbootmybatisprac.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class UserServiceImpl implements UserService {

    @Autowired
    private UserMapper userMapper;

    @Override
    public User findById(Integer id) {
        return userMapper.findById(id);
    }
}
