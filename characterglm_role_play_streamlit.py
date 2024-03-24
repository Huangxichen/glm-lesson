"""
一个简单的demo，调用CharacterGLM实现角色扮演，调用CogView生成图片，调用ChatGLM生成CogView所需的prompt。

依赖：
pyjwt
requests
streamlit
zhipuai
python-dotenv

运行方式：
```bash
streamlit run characterglm_role_play_streamlit.py
```
"""
import os
import itertools
from typing import Iterator, Optional

import streamlit as st
from dotenv import load_dotenv
# 通过.env文件设置环境变量
# reference: https://github.com/theskumar/python-dotenv
load_dotenv()

import api
from api import generate_chat_scene_prompt, generate_role_appearance, get_characterglm_response, generate_cogview_image
from data_types import TextMsg, ImageMsg, TextMsgList, MsgList, filter_text_msg

st.set_page_config(page_title="CharacterGLM API Demo", page_icon="🤖", layout="wide")
debug = os.getenv("DEBUG", "").lower() in ("1", "yes", "y", "true", "t", "on")

def update_api_key(key: Optional[str] = None):
    if debug:
        print(f'update_api_key. st.session_state["API_KEY"] = {st.session_state["API_KEY"]}, key = {key}')
    key = key or st.session_state["API_KEY"]
    if key:
        api.API_KEY = key

# 设置API KEY
api_key = st.sidebar.text_input("API_KEY", value=os.getenv("API_KEY", ""), key="API_KEY", type="password", on_change=update_api_key)
update_api_key(api_key)


# 初始化
if "history" not in st.session_state:
    st.session_state["history"] = []
if "meta" not in st.session_state:
    st.session_state["meta"] = {
        "bot_info": "秦始皇，嬴政，坚定的统一者，才华横溢的君主。他勇猛果敢，统一六国，结束了长期的战争。他推行中央集权，改革法制，建立强大的秦朝。然而，他的暴政也引发了民怨。秦始皇，一个复杂而矛盾的历史人物，对中国历史产生了深远影响。",
        "bot_name": "秦始皇",
        "user_info": "朱元璋，明朝开国皇帝，出身贫寒，曾当过放牛娃和和尚。他勇敢聪明，领导才能出众，领导起义军反抗元朝统治，成功建立了明朝。他推行一系列改革措施，加强中央集权，巩固国家统一。然而，他的执政手段也引发了一些争议。朱元璋，一个传奇而充满智慧的历史人物，对中国历史产生了深远影响。",
        "user_name": "朱元璋"
    }

if "role2_history" not in st.session_state:
    st.session_state["role2_history"] = []
if "meta2" not in st.session_state:
    st.session_state["meta2"] = {
        "bot_info": "朱元璋，明朝开国皇帝，出身贫寒，曾当过放牛娃和和尚。他勇敢聪明，领导才能出众，领导起义军反抗元朝统治，成功建立了明朝。他推行一系列改革措施，加强中央集权，巩固国家统一。然而，他的执政手段也引发了一些争议。朱元璋，一个传奇而充满智慧的历史人物，对中国历史产生了深远影响。",
        "bot_name": "朱元璋",
        "user_info": "秦始皇，嬴政，坚定的统一者，才华横溢的君主。他勇猛果敢，统一六国，结束了长期的战争。他推行中央集权，改革法制，建立强大的秦朝。然而，他的暴政也引发了民怨。秦始皇，一个复杂而矛盾的历史人物，对中国历史产生了深远影响。",
        "user_name": "秦始皇"
    }

def init_session():
    st.session_state["history"] = []
    st.session_state["role2_history"] = []


# 4个输入框，设置meta的4个字段
meta_labels = {
    "bot_name": "角色名",
    "user_name": "用户名", 
    "bot_info": "角色人设",
    "user_info": "用户人设"
}

# 2x2 layout
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.text_input(label="角色名", value=st.session_state["meta"]["bot_name"], key="bot_name", on_change=lambda : st.session_state["meta"].update(bot_name=st.session_state["bot_name"]), help="模型所扮演的角色的名字，不可以为空")
        st.text_area(label="角色人设", value=st.session_state["meta"]["bot_info"], key="bot_info", on_change=lambda : st.session_state["meta"].update(bot_info=st.session_state["bot_info"]), help="角色的详细人设信息，不可以为空")
        
    with col2:
        st.text_input(label="用户名", value=st.session_state["meta"]["user_name"], key="user_name", on_change=lambda : st.session_state["meta"].update(user_name=st.session_state["user_name"]), help="用户的名字，默认为用户")
        st.text_area(label="用户人设", value=st.session_state["meta"]["user_info"], key="user_info", on_change=lambda : st.session_state["meta"].update(user_info=st.session_state["user_info"]), help="用户的详细人设信息，可以为空")


def verify_meta() -> bool:
    # 检查`角色名`和`角色人设`是否空，若为空，则弹出提醒
    if st.session_state["meta"]["bot_name"] == "" or st.session_state["meta"]["bot_info"] == "":
        st.error("角色名和角色人设不能为空")
        return False
    else:
        return True

def save_chat():
    with open('dialogue.txt', 'w') as f:
        for line in st.session_state["history"]:
            f.write(f"{line}\n")

button_labels = {
    "clear_history": "清空对话历史",
    "save_chat": "保存对话"
}
if debug:
    button_labels.update({
        "show_api_key": "查看API_KEY",
        "show_meta": "查看meta",
        "show_history": "查看历史"
    })

# 在同一行排列按钮
with st.container():
    n_button = len(button_labels)
    cols = st.columns(n_button)
    button_key_to_col = dict(zip(button_labels.keys(), cols))

    with button_key_to_col["clear_history"]:
        clear_history = st.button(button_labels["clear_history"], key="clear_history")
        if clear_history:
            init_session()
            st.rerun()

    with button_key_to_col["save_chat"]:
        st.button(label="保存对话", key="saveBtn", on_click=save_chat)

    if debug:
        with button_key_to_col["show_api_key"]:
            show_api_key = st.button(button_labels["show_api_key"], key="show_api_key")
            if show_api_key:
                print(f"API_KEY = {api.API_KEY}")
        
        with button_key_to_col["show_meta"]:
            show_meta = st.button(button_labels["show_meta"], key="show_meta")
            if show_meta:
                print(f"meta = {st.session_state['meta']}")
        
        with button_key_to_col["show_history"]:
            show_history = st.button(button_labels["show_history"], key="show_history")
            if show_history:
                print(f"history = {st.session_state['history']}")


# 展示对话历史
for msg in st.session_state["history"]:
    if msg["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(msg["content"])
    elif msg["role"] == "image":
        with st.chat_message(name="assistant", avatar="assistant"):
            st.image(msg["image"], caption=msg.get("caption", None))
    else:
        raise Exception("Invalid role")



with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()


def output_stream_response(response_stream: Iterator[str], placeholder):
    content = ""
    for content in itertools.accumulate(response_stream):
        placeholder.markdown(content)
    return content

def start_chat():
    query = st.chat_input("请输入讨论主题")
    if not query:
        return
    else:
        if not verify_meta():
            return
        if not api.API_KEY:
            st.error("未设置API_KEY")

        input_placeholder.markdown(query)
        st.session_state["history"].append(TextMsg({"role": "user", "content": query}))
        st.session_state["role2_history"].append(TextMsg({"role": "assistant", "content": query}))

        response_stream = get_characterglm_response(filter_text_msg(st.session_state["history"]), meta=st.session_state["meta"])
        bot_response = output_stream_response(response_stream, message_placeholder)
        if not bot_response:
            message_placeholder.markdown("生成出错")
            st.session_state["history"].pop()
        else:
            st.session_state["history"].append(TextMsg({"role": "assistant", "content": bot_response}))
            st.session_state["role2_history"].append(TextMsg({"role": "user", "content": bot_response}))
        start_role2_chat()
        st.rerun()

def start_role1_chat():
    if not verify_meta():
        return
    if not api.API_KEY:
        st.error("未设置API_KEY")

    response_stream = get_characterglm_response(filter_text_msg(st.session_state["history"]),
                                                meta=st.session_state["meta"])
    bot_response = output_stream_response(response_stream, message_placeholder)
    if not bot_response:
        message_placeholder.markdown("生成出错")
        st.session_state["history"].pop()
    else:
        st.session_state["history"].append(TextMsg({"role": "assistant", "content": bot_response}))
        st.session_state["role2_history"].append(TextMsg({"role": "user", "content": bot_response}))
    start_role2_chat()

max_turns = 10
index = 0

def start_role2_chat():
    if not verify_meta():
        return
    if not api.API_KEY:
        st.error("未设置API_KEY")

    response_stream = get_characterglm_response(filter_text_msg(st.session_state["role2_history"]),
                                                meta=st.session_state["meta2"])
    bot_response = output_stream_response(response_stream, message_placeholder)
    if not bot_response:
        message_placeholder.markdown("生成出错")
        st.session_state["role2_history"].pop()
    else:
        st.session_state["role2_history"].append(TextMsg({"role": "assistant", "content": bot_response}))
        st.session_state["history"].append(TextMsg({"role": "user", "content": bot_response}))
    global index
    while index <= max_turns:
        index += 1
        start_role1_chat()

    
start_chat()
