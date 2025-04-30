package com.xiaoyu.springbootmybatisprac.mapper;

import com.xiaoyu.springbootmybatisprac.pojo.User;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface UserMapper {

    @Select("SELECT * FROM user WHERE id = #{id}")
    public User findById(Integer id);

}
